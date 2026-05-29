"""
upa_beam_distance.py
--------------------
Peak beam index distance metric for a BS Uniform Planar Array (UPA).

Background
----------
After the UPA DFT codebook is applied (see ``upa_dft_codebook`` in
``inference_FMM_custom.py``), the BS-side beamspace representation is
stored as a *flattened* vector of length::

    N_BS_beams = N_tx_x * N_tx_y

Reshape convention (matches ``np.kron(Ay, Ax)`` used in
``upa_dft_codebook``)::

    flat index k  →  iy = k // N_tx_x ,  ix = k % N_tx_x
    reshape(..., N_BS_beams) → reshape(..., N_tx_y, N_tx_x)
    last two axes: (..., iy, ix)

    iy  — vertical BS beam index   (0 … N_tx_y − 1)
    ix  — horizontal BS beam index (0 … N_tx_x − 1)

Public API
----------
::

    from upa_beam_distance import compute_upa_peak_beam_distance

    results = compute_upa_peak_beam_distance(H_rt, H_gen, N_tx_x=8, N_tx_y=4)
    print(results.mean_distance)
"""
from __future__ import annotations

import numpy as np
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class BeamDistanceResult(NamedTuple):
    """Structured return value of :func:`compute_upa_peak_beam_distance`.

    Attributes
    ----------
    dominant_rt : np.ndarray, shape (N_samples, 2)
        Dominant BS beam index ``(iy_star, ix_star)`` for every RT sample.
    dominant_gen : np.ndarray, shape (N_samples, 2)
        Dominant BS beam index ``(iy_star, ix_star)`` for every Gen sample.
    distances : np.ndarray, shape (N_samples,)
        Euclidean 2-D beam index distance per sample::

            sqrt((iy_RT - iy_Gen)^2 + (ix_RT - ix_Gen)^2)

    mean_distance : float
        Mean of ``distances`` over all samples.
    std_distance : float
        Standard deviation of ``distances`` over all samples.
    """
    dominant_rt:   np.ndarray
    dominant_gen:  np.ndarray
    distances:     np.ndarray
    mean_distance: float
    std_distance:  float


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_upa_peak_beam_distance(
    H_rt:   np.ndarray,
    H_gen:  np.ndarray,
    N_tx_x: int,
    N_tx_y: int,
    eps:    float = 1e-12,
) -> BeamDistanceResult:
    """Compute the sample-wise 2-D peak beam index distance for a BS UPA.

    For each sample ``n`` and each channel type ``q ∈ {RT, Gen}``:

    1. Compute the BS-side power map by summing over UE beams::

           P_q(n, iy, ix) = Σ_j |H_q(n, j, iy, ix)|²

    2. Normalize so the total power per sample equals 1::

           p_q(n, iy, ix) = P_q(n, iy, ix) / (Σ_{iy,ix} P_q(n, iy, ix) + ε)

    3. Find the dominant beam::

           (iy_q*, ix_q*) = argmax_{iy, ix} p_q(n, iy, ix)

    4. Compute the Euclidean 2-D beam index distance::

           d(n) = sqrt((iy_RT* − iy_Gen*)² + (ix_RT* − ix_Gen*)²)

    Parameters
    ----------
    H_rt, H_gen : np.ndarray
        Reference (ray-tracing) and generated beamspace channel matrices.
        Two input formats are accepted:

        **Complex format** (dtype complex):
            shape ``(N_samples, N_UE_beams, N_BS_beams)``

        **Split real/imag format** (dtype real):
            shape ``(N_samples, 2, N_UE_beams, N_BS_beams)``
            where ``H[:, 0, :, :]`` = real part, ``H[:, 1, :, :]`` = imag part.

    N_tx_x : int
        Number of BS antennas along the **horizontal** axis.
    N_tx_y : int
        Number of BS antennas along the **vertical** axis.
        Must satisfy ``N_BS_beams == N_tx_x * N_tx_y``.
    eps : float, optional
        Small value added to power denominators to prevent division by zero.
        Default ``1e-12``.

    Returns
    -------
    BeamDistanceResult
        Named tuple; see :class:`BeamDistanceResult` for field descriptions.

    Raises
    ------
    ValueError
        If ``H_rt`` and ``H_gen`` have different shapes, or if
        ``N_BS_beams != N_tx_x * N_tx_y``.
    """
    # 1. Convert to complex (N_samples, N_UE_beams, N_BS_beams)
    H_rt_c  = _to_complex(H_rt)
    H_gen_c = _to_complex(H_gen)

    # 2. Sanity checks
    _check_inputs(H_rt_c, H_gen_c, N_tx_x, N_tx_y)

    # 3. Reshape flat BS beam dimension into 2-D UPA grid
    #    Output: (N_samples, N_UE_beams, N_tx_y, N_tx_x)
    #    Last two axes are (iy, ix) — see module docstring for convention.
    H_rt_upa  = _reshape_bs_to_upa(H_rt_c,  N_tx_x, N_tx_y)
    H_gen_upa = _reshape_bs_to_upa(H_gen_c, N_tx_x, N_tx_y)

    # 4. Normalized BS-side power distribution
    #    Output: (N_samples, N_tx_y, N_tx_x)
    p_rt  = _bs_power_distribution(H_rt_upa,  eps=eps)
    p_gen = _bs_power_distribution(H_gen_upa, eps=eps)

    # 5. Dominant beam indices
    #    Output: (N_samples, 2)  — each row is (iy_star, ix_star)
    idx_rt  = _dominant_beam_index(p_rt)
    idx_gen = _dominant_beam_index(p_gen)

    # 6. Euclidean 2-D beam index distance per sample
    #    Output: (N_samples,)
    diff      = (idx_rt - idx_gen).astype(float)   # (N_samples, 2)
    distances = np.sqrt(np.sum(diff ** 2, axis=1)) # (N_samples,)

    assert distances.shape == (H_rt_c.shape[0],), (
        f"distances shape mismatch: expected ({H_rt_c.shape[0]},), "
        f"got {distances.shape}"
    )

    return BeamDistanceResult(
        dominant_rt=idx_rt,
        dominant_gen=idx_gen,
        distances=distances,
        mean_distance=float(np.mean(distances)),
        std_distance=float(np.std(distances)),
    )


# ---------------------------------------------------------------------------
# Helper: real/imag → complex
# ---------------------------------------------------------------------------

def _to_complex(H: np.ndarray) -> np.ndarray:
    """Convert a beamspace channel array to complex format.

    Parameters
    ----------
    H : np.ndarray
        Either:

        * Complex array of shape ``(N_samples, N_UE_beams, N_BS_beams)``, or
        * Real array of shape ``(N_samples, 2, N_UE_beams, N_BS_beams)``
          where ``H[:, 0, :, :]`` = real part and ``H[:, 1, :, :]`` = imag part.

    Returns
    -------
    np.ndarray, complex128
        Shape ``(N_samples, N_UE_beams, N_BS_beams)``.
    """
    if np.iscomplexobj(H):
        if H.ndim != 3:
            raise ValueError(
                f"Complex H must have 3 dimensions (N_samples, N_UE_beams, N_BS_beams), "
                f"got shape {H.shape}."
            )
        return H.astype(np.complex128)

    # Split real/imag format: (N_samples, 2, N_UE_beams, N_BS_beams)
    if H.ndim == 4 and H.shape[1] == 2:
        return (H[:, 0, :, :] + 1j * H[:, 1, :, :]).astype(np.complex128)

    raise ValueError(
        f"Unrecognised H shape {H.shape}. "
        "Expected (N_samples, N_UE_beams, N_BS_beams) [complex dtype] or "
        "(N_samples, 2, N_UE_beams, N_BS_beams) [split real/imag, real dtype]."
    )


# ---------------------------------------------------------------------------
# Helper: reshape flat BS dimension → UPA grid
# ---------------------------------------------------------------------------

def _reshape_bs_to_upa(
    H_c:    np.ndarray,
    N_tx_x: int,
    N_tx_y: int,
) -> np.ndarray:
    """Reshape the flattened BS beam axis into a 2-D UPA grid.

    Reshape convention (matches ``np.kron(Ay, Ax)`` Kronecker ordering
    used in ``upa_dft_codebook``)::

        flat index k  →  iy = k // N_tx_x ,  ix = k % N_tx_x
        last two output axes: (iy, ix)

    Parameters
    ----------
    H_c : np.ndarray, complex
        Shape ``(N_samples, N_UE_beams, N_BS_beams)``.
    N_tx_x : int
        Number of BS antennas along the horizontal axis (fast-varying index).
    N_tx_y : int
        Number of BS antennas along the vertical axis (slow-varying index).

    Returns
    -------
    np.ndarray, complex
        Shape ``(N_samples, N_UE_beams, N_tx_y, N_tx_x)``.
        Axes: sample, UE beam, **iy** (vertical), **ix** (horizontal).
    """
    N_samples, N_UE_beams, _ = H_c.shape
    # NumPy default (C order): last axis varies fastest, matching the
    # np.kron(Ay, Ax) convention where ix is the fast-varying index.
    return H_c.reshape(N_samples, N_UE_beams, N_tx_y, N_tx_x)


# ---------------------------------------------------------------------------
# Helper: BS-side power distribution
# ---------------------------------------------------------------------------

def _bs_power_distribution(
    H_upa: np.ndarray,
    eps:   float = 1e-12,
) -> np.ndarray:
    """Compute the normalized BS-side beamspace power distribution.

    Sums ``|H|²`` over the UE-side beam dimension only (axis 1), then
    normalizes each sample so the total power sums to 1::

        P(n, iy, ix) = Σ_j |H_upa(n, j, iy, ix)|²
        p(n, iy, ix) = P(n, iy, ix) / (Σ_{iy,ix} P(n, iy, ix) + ε)

    Parameters
    ----------
    H_upa : np.ndarray, complex
        Shape ``(N_samples, N_UE_beams, N_tx_y, N_tx_x)``.
    eps : float
        Added to denominator to prevent division by zero.

    Returns
    -------
    np.ndarray, float64
        Normalized power map of shape ``(N_samples, N_tx_y, N_tx_x)``.
        Axes -2, -1 correspond to ``(iy, ix)``.
    """
    # Sum |H|² over UE beams → (N_samples, N_tx_y, N_tx_x)
    # NOTE: we sum over the UE dimension ONLY; the BS-side grid is preserved.
    power = np.sum(np.abs(H_upa) ** 2, axis=1)  # (N_samples, N_tx_y, N_tx_x)

    # Normalize per sample: total power over entire beam grid equals 1
    total = np.sum(power, axis=(1, 2), keepdims=True) + eps  # (N_samples, 1, 1)
    return power / total


# ---------------------------------------------------------------------------
# Helper: dominant beam index
# ---------------------------------------------------------------------------

def _dominant_beam_index(power_2d: np.ndarray) -> np.ndarray:
    """Return the 2-D index of the dominant beam for each sample.

    Parameters
    ----------
    power_2d : np.ndarray, float
        Normalized power map of shape ``(N_samples, N_tx_y, N_tx_x)``.
        Axes -2, -1 are ``(iy, ix)``.

    Returns
    -------
    np.ndarray, int
        Shape ``(N_samples, 2)``.  Each row is ``[iy_star, ix_star]``.
    """
    N_samples, N_tx_y, N_tx_x = power_2d.shape

    # Flatten the 2-D beam grid → (N_samples, N_tx_y * N_tx_x), find argmax
    flat_idx = np.argmax(power_2d.reshape(N_samples, -1), axis=1)  # (N_samples,)

    # Convert flat index back to 2-D (iy, ix)
    iy_star = flat_idx // N_tx_x  # vertical beam index   (slow axis)
    ix_star = flat_idx  % N_tx_x  # horizontal beam index (fast axis)

    return np.stack([iy_star, ix_star], axis=1)  # (N_samples, 2)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _check_inputs(
    H_rt_c:  np.ndarray,
    H_gen_c: np.ndarray,
    N_tx_x:  int,
    N_tx_y:  int,
) -> None:
    """Validate channel arrays before computing the beam distance metric.

    Raises
    ------
    ValueError
        On shape mismatch or incompatible antenna dimensions.
    """
    if H_rt_c.shape != H_gen_c.shape:
        raise ValueError(
            f"RT and Gen channels must have identical shapes; "
            f"got RT={H_rt_c.shape} vs Gen={H_gen_c.shape}."
        )
    N_BS_beams = H_rt_c.shape[2]
    expected   = N_tx_x * N_tx_y
    if N_BS_beams != expected:
        raise ValueError(
            f"N_BS_beams={N_BS_beams} must equal N_tx_x × N_tx_y = "
            f"{N_tx_x} × {N_tx_y} = {expected}."
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_tests(N_tx_x: int = 8, N_tx_y: int = 4) -> None:
    """Quick sanity-checks — run with ``python upa_beam_distance.py``."""
    rng = np.random.default_rng(42)
    N_samples  = 16
    N_UE_beams = 4
    N_BS_beams = N_tx_x * N_tx_y  # 32

    # ------------------------------------------------------------------
    # Test 1 — complex format, identical RT == Gen → distance must be 0
    # ------------------------------------------------------------------
    H = (rng.standard_normal((N_samples, N_UE_beams, N_BS_beams))
         + 1j * rng.standard_normal((N_samples, N_UE_beams, N_BS_beams)))

    res = compute_upa_peak_beam_distance(H, H, N_tx_x, N_tx_y)

    assert res.distances.shape   == (N_samples,),  f"Wrong shape: {res.distances.shape}"
    assert res.dominant_rt.shape == (N_samples, 2), f"Wrong shape: {res.dominant_rt.shape}"
    assert res.dominant_gen.shape== (N_samples, 2), f"Wrong shape: {res.dominant_gen.shape}"
    assert np.all(res.distances == 0.0), f"Expected all-zero distances; got {res.distances}"
    print("[PASS] Test 1: complex format, identical channels → distance = 0.")

    # ------------------------------------------------------------------
    # Test 2 — split real/imag format, identical RT == Gen
    # ------------------------------------------------------------------
    H_split = np.stack([H.real, H.imag], axis=1)  # (N_samples, 2, N_UE, N_BS)
    res2 = compute_upa_peak_beam_distance(H_split, H_split, N_tx_x, N_tx_y)
    assert np.all(res2.distances == 0.0), "Split-format identical channels should give distance 0."
    print("[PASS] Test 2: split real/imag format, identical channels → distance = 0.")

    # ------------------------------------------------------------------
    # Test 3 — independent random RT and Gen → non-negative distances
    # ------------------------------------------------------------------
    H_gen = (rng.standard_normal((N_samples, N_UE_beams, N_BS_beams))
             + 1j * rng.standard_normal((N_samples, N_UE_beams, N_BS_beams)))
    res3 = compute_upa_peak_beam_distance(H, H_gen, N_tx_x, N_tx_y)
    assert res3.distances.shape == (N_samples,)
    assert np.all(res3.distances >= 0.0)
    print(f"[PASS] Test 3: random RT vs Gen — "
          f"mean distance = {res3.mean_distance:.4f}, std = {res3.std_distance:.4f}.")

    # ------------------------------------------------------------------
    # Test 4 — wrong N_BS_beams → ValueError
    # ------------------------------------------------------------------
    try:
        bad = rng.standard_normal((N_samples, N_UE_beams, N_BS_beams + 1)) + 0j
        compute_upa_peak_beam_distance(bad, bad, N_tx_x, N_tx_y)
        raise AssertionError("Expected ValueError for mismatched N_BS_beams.")
    except ValueError:
        print("[PASS] Test 4: mismatched N_BS_beams raises ValueError.")

    # ------------------------------------------------------------------
    # Test 5 — RT and Gen with different shapes → ValueError
    # ------------------------------------------------------------------
    try:
        H_bad = (rng.standard_normal((N_samples + 1, N_UE_beams, N_BS_beams))
                 + 0j)
        compute_upa_peak_beam_distance(H, H_bad, N_tx_x, N_tx_y)
        raise AssertionError("Expected ValueError for shape mismatch.")
    except ValueError:
        print("[PASS] Test 5: shape mismatch between RT and Gen raises ValueError.")

    print("\nAll tests passed.")


if __name__ == "__main__":
    _run_tests()
