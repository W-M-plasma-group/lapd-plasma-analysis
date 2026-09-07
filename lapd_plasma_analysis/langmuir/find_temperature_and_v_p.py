"""
Drop-in region-finder + breakpoint T_e extraction for Langmuir I-V sweeps.

Replaces the tanh-derivative window finder in get_t_e_spline with:
  (1) floor removal + noise yardstick        -> floor_and_noise()
  (2) sliding Theil-Sen local-slope plateau  -> theilsen_plateau()   [rough region]
  (3) two-line breakpoint optimizer          -> breakpoint_fit()     [exact knee + Vp]

Design constraint honored: T_e is ALWAYS 1/slope of a straight line fit to the
real ln(I) data inside the bracket. The models only locate the region; they
never set the slope. Theil-Sen is used for the reported slope so one-sided
log-noise spikes cannot tilt it.

All *_core functions take plain float numpy arrays (bias in V, current in A).
The units-aware wrapper get_t_e_breakpoint() matches your existing signature.
"""
import numpy as np
from scipy.stats import theilslopes
from scipy.ndimage import uniform_filter1d


# ---------------------------------------------------------------------------
# Step 0 (optional): estimate the floating potential from the sweep itself
# ---------------------------------------------------------------------------
def vfloat_from_sweep(V, I, smooth_pts=31, floor_frac=0.40, snr_k=3.0):
    """Estimate V_float = electron-onset zero-crossing, robust to floor noise.

    Use this only if you don't already measure V_float. It smooths the current,
    measures the floor noise on the deep-negative region, and returns the last
    negative->positive crossing that occurs before the current climbs decisively
    (> snr_k * floor RMS) above the floor. That "before the real rise" guard
    stops it latching onto a stray zero-crossing far out in ion saturation.

    V, I : float arrays (bias [V], current [A]), sorted by V ascending.
    """
    Is = uniform_filter1d(I, smooth_pts)
    floor = V < np.percentile(V, 100*floor_frac)
    rms = np.std(Is[floor]) if floor.any() else np.std(Is)
    crossings = np.where((Is[:-1] < 0) & (Is[1:] >= 0))[0]
    if crossings.size == 0:
        return V[np.argmin(np.abs(Is))]
    onset = np.where(Is > snr_k * rms)[0]
    if onset.size:
        before = crossings[crossings <= onset[0]]
        if before.size:
            return V[before[-1]]
    return V[crossings[-1]]


# ---------------------------------------------------------------------------
# Step 1: remove ion floor, measure the RMS noise around it (the yardstick)
# ---------------------------------------------------------------------------
def floor_and_noise_core(V, I, vf, ion_current=None, floor_frac=0.30):
    """
    Subtract the ion current and measure the noise around the ion floor.

    V, I         : float arrays, bias [V] and current [A], sorted by V ascending
    vf           : floating-potential bias [V]
    ion_current  : precomputed ion current array [A] (from your get_ion_current).
                   If None, a robust (Theil-Sen) line is fit to the deep-negative
                   region and used as the floor.
    floor_frac   : fraction of the most-negative-bias points used to measure noise.

    Returns
    -------
    I_e      : electron current = I - ion_current
    floor_rms: RMS scatter of I about the ion floor in the deep-negative region
    """
    if ion_current is None:
        n_floor = max(5, int(floor_frac * V.size))
        sel = slice(0, n_floor)                       # most-negative bias points
        res = theilslopes(I[sel], V[sel])             # slope, intercept, lo, hi
        a, b = res[1], res[0]                          # intercept, slope  (I = a + b*V)
        ion_current = a + b * V
    I_e = I - ion_current

    # noise = scatter of the raw current about the floor line in the floor window
    n_floor = max(5, int(floor_frac * V.size))
    floor_resid = I_e[:n_floor] - np.median(I_e[:n_floor])
    floor_rms = np.sqrt(np.mean(floor_resid**2))
    return I_e, floor_rms


# ---------------------------------------------------------------------------
# Step 2 helper: mask the floor, take the log of survivors only (no offset shift)
# ---------------------------------------------------------------------------
def masked_log_core(V, I_e, floor_rms, vf, snr_k=4.0):
    """
    Keep only points with electron current safely above the noise, and points
    at/above the floating potential (electron branch). Return ln(I_e) for those.

    Returns V_keep, lnI_keep, keep_mask  (keep_mask indexes the input arrays)
    """
    keep = (I_e > snr_k * floor_rms) & (V >= vf)
    return V[keep], np.log(I_e[keep]), keep


# ---------------------------------------------------------------------------
# Step 3: sliding Theil-Sen local slope -> find the constant-slope plateau
# ---------------------------------------------------------------------------

def _theil(y, x, cap=200):
    """Theil-Sen slope/intercept on at most `cap` evenly-spaced points.

    scipy's theilslopes is O(n^2) (every pairwise slope). On the long
    retarding/saturation segments that dominates runtime, but the robust slope
    is insensitive to dropping to ~150-200 well-spread points (Te changes
    <0.2%). For n <= cap this is identical to calling theilslopes directly.
    """
    x = np.asarray(x); y = np.asarray(y)
    n = x.size
    if n > cap:
        idx = np.linspace(0, n - 1, cap).round().astype(int)
        x, y = x[idx], y[idx]
    return theilslopes(y, x)

def theilsen_plateau_core(V, lnI, win_pts=9, tol_mad=3.0, min_efold=1.5):
    """
    Slide a Theil-Sen slope along ln(I) vs V and return the contiguous stretch
    where the local slope sits on a stable plateau (the retarding region).

    The retarding region is the STEEPEST stable slope (saturation is shallower,
    the floor is masked out), so we anchor at the max-slope point and walk
    outward while the slope stays within tol_mad robust-deviations of the local
    plateau value. This avoids latching onto the long shallow saturation plateau.
    """
    n = V.size
    if n < win_pts + 2:
        return dict(lo=0, hi=n-1, m_plateau=np.nan, m_mad=np.nan, efold=0.0, ok=False)

    half = win_pts // 2
    m = np.full(n, np.nan)
    for i in range(half, n - half):
        s = slice(i-half, i+half+1)
        m[i] = theilslopes(lnI[s], V[s])[0]
    valid = np.isfinite(m) & (m > 0)
    if valid.sum() < 3:
        return dict(lo=0, hi=n-1, m_plateau=np.nan, m_mad=np.nan, efold=0.0, ok=False)

    m_mad = np.median(np.abs(m[valid] - np.median(m[valid]))) + 1e-12

    # anchor: steepest local slope (mid-retarding). Take median in its neighborhood.
    anchor = np.nanargmax(np.where(valid, m, -np.inf))
    lo0, hi0 = max(0, anchor-half), min(n-1, anchor+half)
    m_plateau = np.median(m[lo0:hi0+1])

    # walk outward while slope stays within band of the retarding plateau value
    band = tol_mad * m_mad
    lo = anchor
    while lo-1 >= 0 and np.isfinite(m[lo-1]) and abs(m[lo-1]-m_plateau) < band:
        lo -= 1
    hi = anchor
    while hi+1 < n and np.isfinite(m[hi+1]) and abs(m[hi+1]-m_plateau) < band:
        hi += 1

    efold = lnI[hi] - lnI[lo]
    return dict(lo=lo, hi=hi, m_plateau=np.median(m[lo:hi+1]),
                m_mad=m_mad, efold=efold, ok=(efold >= min_efold))


# ---------------------------------------------------------------------------
# Step 4-5: two-line breakpoint optimizer -> knee, T_e, V_p
# ---------------------------------------------------------------------------
def _ss_line(x, y):
    """least-squares line fit, return (slope, intercept, SSE)."""
    m, b = np.polyfit(x, y, 1)
    sse = np.sum((y - (m*x + b))**2)
    return m, b, sse


def _prefix_segments(x, y):
    """
    Return a closure seg(lo, hi) giving the least-squares slope, intercept, and
    SSE of a straight line over each contiguous segment x[lo:hi], computed in
    O(1) per segment from cumulative sums. lo/hi may be integer arrays, so
    every candidate breakpoint split is evaluated in a single vectorized pass
    instead of one polyfit per corner. Mathematically identical to _ss_line.
    """
    def pref(a):
        return np.concatenate([[0.0], np.cumsum(a)])
    Sx, Sy = pref(x), pref(y)
    Sxx, Syy, Sxy = pref(x*x), pref(y*y), pref(x*y)

    def seg(lo, hi):
        lo = np.asarray(lo); hi = np.asarray(hi)
        cnt = (hi - lo).astype(float)
        sx = Sx[hi]-Sx[lo]; sy = Sy[hi]-Sy[lo]
        sxx = Sxx[hi]-Sxx[lo]; syy = Syy[hi]-Syy[lo]; sxy = Sxy[hi]-Sxy[lo]
        with np.errstate(divide="ignore", invalid="ignore"):
            sxx_c = sxx - sx*sx/cnt
            sxy_c = sxy - sx*sy/cnt
            syy_c = syy - sy*sy/cnt
            m = sxy_c/sxx_c
            b = (sy - m*sx)/cnt
            sse = np.maximum(syy_c - sxy_c*sxy_c/sxx_c, 0.0)
        return m, b, sse
    return seg


def _trim_retarding_endpoint(V, lnI, k, endpoint_tol, min_seg=4,
                             max_trim_frac=0.4):
    """
    After the breakpoint scan picks corner index k (retarding = points [0:k]),
    the rounded knee can leave the last few retarding points curling BELOW the
    straight exponential line -- the corner sits slightly INTO the bend, which
    flattens the fitted slope and inflates T_e.

    Iteratively drop trailing retarding points that fall below the line by more
    than endpoint_tol * sigma, where sigma is the scaled MAD of the current
    residuals. Smaller endpoint_tol = stricter (trims more). Stops when the last
    point is back on the line, or when the retarding segment would shrink past
    min_seg or below (1 - max_trim_frac) of its original length.

    Returns the new retarding endpoint index (<= k). Only the retarding line is
    affected; the saturation segment and its line are unchanged.

    The drop decision uses a fast ordinary-least-squares line (O(n)), NOT the
    robust Theil-Sen fit -- deciding which endpoint curls below the trend does
    not need robustness, and calling theilslopes (O(n^2)) inside this loop makes
    it ~30x slower. The final reported slope is still refit robustly by the
    caller once k_ret is fixed.
    """
    k_ret = k
    k_floor = max(min_seg, int(k * (1.0 - max_trim_frac)))
    for _ in range(k):
        if k_ret <= k_floor:
            break
        x = V[:k_ret]; y = lnI[:k_ret]
        m, b = np.polyfit(x, y, 1)               # fast OLS, O(n)
        res = y - (m*x + b)
        sigma = 1.4826*np.median(np.abs(res - np.median(res)))
        if not np.isfinite(sigma) or sigma <= 0:
            break
        if res[-1] < -endpoint_tol*sigma:        # last point curls below -> drop
            k_ret -= 1
        else:
            break
    return k_ret


def breakpoint_fit_core(V, lnI, min_seg=4, robust_Te=True, sat_window_V=None,
                        endpoint_tol=None):
    """
    Fit two straight lines (retarding below, saturation above) meeting at a
    breakpoint, over the whole above-noise electron branch. Choose the
    breakpoint with the smallest TOTAL fit error, subject to the physical
    constraint that saturation is shallower than the retarding rise.

    sat_window_V : float or None
        If set, the saturation segment is restricted to points within this many
        volts ABOVE the knee. This is done as a two-pass fit: an unrestricted
        pass locates the knee, then all data above (knee + sat_window_V) are
        dropped and the fit is repeated. Use it when the saturation branch is
        noisy or multivalued far from the knee (e.g. up/down-sweep fan-out) --
        the saturation line only needs a short lever arm just past the knee to
        fix the V_p intersection, and far-field scatter otherwise inflates the
        saturation error and can falsely trip the "is there a knee" test.
        The retarding region (which sets T_e) is never touched by this cap.

    Returns dict with:
      k, V_knee, Te, m_ret, b_ret, m_sat, b_sat, V_p, sse, sse_single,
      efold, sharpness, ok
    """
    n = V.size
    out = dict(ok=False)
    if n < 2*min_seg:
        return out

    # Two-pass saturation-window cap: locate knee on full data, then drop the
    # far saturation tail and refit once on the capped range.
    if sat_window_V is not None:
        first = breakpoint_fit_core(V, lnI, min_seg=min_seg,
                                    robust_Te=robust_Te, sat_window_V=None,
                                    endpoint_tol=endpoint_tol)
        if not first["ok"]:
            return first
        cap = V <= first["V_knee"] + sat_window_V
        if cap.sum() < 2 * min_seg:  # not enough room above knee; keep full fit
            return first
        return breakpoint_fit_core(V[cap], lnI[cap], min_seg=min_seg,
                                   robust_Te=robust_Te, sat_window_V=None,
                                   endpoint_tol=endpoint_tol)

    # Vectorized breakpoint scan via prefix sums: evaluate the left (retarding)
    # and right (saturation) line fit for EVERY candidate corner in one pass.
    # Identical result to the per-corner polyfit loop, ~30x faster.
    seg = _prefix_segments(V, lnI)
    _, _, sse_single = seg(np.array([0]), np.array([n]))  # single-line SSE
    sse_single = float(sse_single[0])

    ks = np.arange(min_seg, n - min_seg + 1)             # k = first idx of UPPER seg
    zeros = np.zeros_like(ks)
    mL, bL, ssL = seg(zeros, ks)                          # retarding
    mR, bR, ssR = seg(ks, np.full_like(ks, n))            # saturation
    tot = ssL + ssR
    valid = (mL > 0) & (mR < mL) & (mR >= 0) & np.isfinite(tot)      # rise; sat shallower
    if not valid.any():
        return out
    tv = np.where(valid, tot, np.inf)
    order = np.argsort(tv)
    bi = int(order[0])
    sse = float(tv[bi]); k = int(ks[bi])
    mR_k, bR_k = float(mR[bi]), float(bR[bi])

    # sharpness: 2nd-best valid split ratio (from the same array, free)
    second = tv[order[1]] if (order.size > 1 and np.isfinite(tv[order[1]])) else np.inf
    sharpness = (second/sse - 1.0) if sse > 0 else np.inf

    # Optional endpoint trim: drop trailing retarding points that curl below the
    # line (corner sat slightly inside the rounded knee), which otherwise
    # flattens the slope and inflates T_e. Only the retarding line moves; the
    # knee index k and the saturation line are unchanged.
    k_ret = k
    if endpoint_tol is not None:
        k_ret = _trim_retarding_endpoint(V, lnI, k, endpoint_tol, min_seg=min_seg)

    # reported T_e: refit the LOWER segment robustly (Theil-Sen) on real data
    if robust_Te:
        if robust_Te:
            ts = _theil(lnI[:k_ret], V[:k_ret])  # was: theilslopes(lnI[:k_ret], V[:k_ret])
            m_ret, b_ret = ts[0], ts[1]
    else:
        m_ret, b_ret = float(mL[bi]), float(bL[bi])

    Te = 1.0 / m_ret
    V_p = (bR_k - b_ret) / (m_ret - mR_k)  # line intersection (log space)
    V_knee = V[k]
    efold = lnI[k_ret - 1] - lnI[0]

    out.update(k=k, k_ret=k_ret, V_knee=V_knee, Te=Te, m_ret=m_ret, b_ret=b_ret,
               m_sat=mR_k, b_sat=bR_k, V_p=V_p, sse=sse, sse_single=sse_single,
               efold=efold, sharpness=sharpness, ok=True)
    return out


# ---------------------------------------------------------------------------
# Two-population split: recover a distinct HOT electron segment above the knee
# ---------------------------------------------------------------------------
def two_population_split_core(V, lnI, k_knee, min_seg=5, sat_window_V=None,
                             robust_Te=True):
    """
    Detect a second, hotter electron population sitting between the primary
    (cold/hot) knee and the electron-saturation shelf.

    The primary two-line fit in breakpoint_fit_core locks onto the SHARPEST
    bend in the semilog branch. In a two-temperature sweep that bend is the
    cold/hot corner, so the primary fit reports the COLD slope and places V_p
    at the cold/hot corner -- the hotter population (which actually sets the
    bulk electron temperature) is swallowed into what the primary fit calls
    "saturation".

    This routine re-runs the same two-line breakpoint fit on the branch ABOVE
    the primary knee (V[k_knee:]). If that branch splits into a steep lower
    segment (the hot population) and a flat upper segment (true saturation),
    the lower segment's slope gives the hot T_e and the intersection is the
    true plasma potential.

    Returns dict(ok, k2 (absolute index of hot/sat corner in V), m_hot, b_hot,
    Te_hot, m_sat, b_sat, V_p, V_hotsat, efold_hot, sse, sse_single) or
    dict(ok=False) when there is no distinct hot segment (clean single
    population: the branch above the knee is flat and the second fit fails its
    slope-ordering / knee guards).
    """
    Vu, lnu = V[k_knee:], lnI[k_knee:]
    if Vu.size < 2 * min_seg:
        return dict(ok=False)
    second = breakpoint_fit_core(Vu, lnu, min_seg=min_seg,
                                 robust_Te=robust_Te, sat_window_V=sat_window_V)
    if not second["ok"]:
        return dict(ok=False)
    return dict(ok=True, k2=k_knee + second["k"],
                m_hot=second["m_ret"], b_hot=second["b_ret"],
                Te_hot=second["Te"], m_sat=second["m_sat"],
                b_sat=second["b_sat"], V_p=second["V_p"],
                V_hotsat=second["V_knee"], efold_hot=second["efold"],
                sse=second["sse"], sse_single=second["sse_single"])


# ---------------------------------------------------------------------------
# Units-aware wrapper: matches get_t_e_spline(sorted_bias, sorted_current,
# v_f_bias, ...) and astropy-Quantity conventions. Drop-in.
# ---------------------------------------------------------------------------
import astropy.units as u


def _refine_vp(V, lnI, k_corner, m_ret, b_ret, flat_frac=0.20, min_seg=6):
    """Plasma-potential refinement for the single-population case.

    The breakpoint corner marks where the retarding exponential *starts* to
    roll over, which in a magnetized device with a sloped/rounded saturation
    (electron sat. limited to ~10-20x ion sat., knee indistinct) sits BELOW the
    true plasma potential. The physical V_p is the intersection of the retarding
    line with the electron-saturation line (Pace/Chen two-line construction).

    The default `sat_window_V` fit takes the saturation slope from a fixed
    window just above the corner, which is still inside the rounded transition
    and biases V_p low. Here we instead locate where the log-current is GENUINELY
    flat -- local Theil-Sen slope < flat_frac * m_ret -- scanning upward from the
    corner, fit the saturation line over that flat tail, and intersect.

    Returns (V_p, m_sat, b_sat, sat_lo) or None if no flat tail is found (then
    the caller keeps the original corner-based V_p). Te is never touched.
    """
    n = V.size
    if k_corner >= n - min_seg:
        return None
    thr = flat_frac * m_ret
    step = 2.0
    Vc = V[k_corner]
    lo = Vc
    flat_lo = None
    while lo < V[-1] - 4.0:
        m = (V >= lo) & (V < lo + 4.0)
        if m.sum() >= min_seg:
            s = theilslopes(lnI[m], V[m])[0]
            if s < thr:
                flat_lo = lo
                break
        lo += step
    if flat_lo is None:
        return None
    sat = V >= flat_lo
    if sat.sum() < min_seg:
        return None
    s_sat, i_sat = _theil(lnI[sat], V[sat])[:2]
    denom = (m_ret - s_sat)
    if abs(denom) < 1e-9:
        return None
    V_p = (i_sat - b_ret) / denom
    if not (Vc <= V_p <= V[-1]):
        return None
    return float(V_p), float(s_sat), float(i_sat), float(flat_lo)


def get_t_e_breakpoint(sorted_bias, sorted_current, v_f_bias,
                       ion_current=None, snr_k=4.0, win_pts=9, min_seg=5,
                       efold_min=1.0, knee_ratio=0.3,
                       r2_high=0.90, efold_high=1.5, sat_window_V=12.0,
                       endpoint_tol=1.5,
                       two_pop=True, hot_ratio=1.3, hot_ratio_max=4.0,
                       hot_efold_min=0.5,
                       refine_vp=True, batch_mode=False):
    """Region-find-then-line-fit T_e extraction (breakpoint method).

    batch_mode : if True, skip the sliding-window Theil-Sen plateau finder
        (informational only -- it never gates) to save ~0.2 s/sweep. The
        Te_plateau / plateau_agree / plateau_ok flags are then omitted. Use for
        large campaigns where you only need Te, V_p, quality, and fit_region.

    Tiered quality gate (see breakpoint fit section for rationale):
      * knee_ratio  : two-line SSE must be < knee_ratio * single-line SSE, else
                      the knee is false/displaced and T_e is rejected. This is
                      the structural discriminator.
      * efold_min   : minimum retarding lever arm (e-foldings) to report at all.
      * r2_high, efold_high : a PASS with r2>=r2_high AND efold>=efold_high is
                      tagged quality='high'; otherwise 'medium'. These GRADE,
                      they do not reject -- a noisy-but-real knee still returns
                      a T_e (flagged 'medium') rather than being nulled.
      * hot_ratio_max : Te_hot/Te_cold must be <= this to be PHYSICAL. A ratio
                        far above the bulk (default 4x) is the rounded
                        saturation roll-off misread as a hot Maxwellian, not a
                        real population -- rejected back to single-population.
                        Real hot cores here run ~2-3x; the spurious edge
                        artifacts run >10x, so the bound separates them cleanly
                        without needing to know the probe position.

    Returns a dict of Quantities + quality flags. Te=None (with flags) only when
    the knee is not real or the lever arm is too short, so batch callers can drop
    the genuinely unusable sweeps while keeping noisy-but-valid ones.
    """
    V  = sorted_bias.to(u.V).value.astype(float)
    I  = sorted_current.to(u.A).value.astype(float)
    vf = v_f_bias.to(u.V).value

    ic = None if ion_current is None else ion_current.to(u.A).value.astype(float)
    I_e, floor_rms = floor_and_noise_core(V, I, vf, ion_current=ic)
    Vk, lnI, keep  = masked_log_core(V, I_e, floor_rms, vf, snr_k=snr_k)

    flags = dict(floor_rms=floor_rms, n_kept=int(keep.sum()))
    if Vk.size < 2*min_seg:
        flags["fail"] = "too_few_points"
        return dict(Te=None, V_p=None, flags=flags)

    bp = breakpoint_fit_core(Vk, lnI, min_seg=min_seg, sat_window_V=sat_window_V,
                             endpoint_tol=endpoint_tol)

    if not bp["ok"]:
        flags["fail"] = "no_valid_breakpoint"
        return dict(Te=None, V_p=None, flags=flags)

    xk, yk = Vk[:bp.get("k_ret", bp["k"])], lnI[:bp.get("k_ret", bp["k"])]
    yhat = bp["m_ret"]*xk + bp["b_ret"]
    ss_res = np.sum((yk-yhat)**2); ss_tot = np.sum((yk-yk.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0

    knee_real = bool(bp["sse"] < knee_ratio*bp["sse_single"])
    flags.update(r2=r2, efold=bp["efold"], sharpness=bp["sharpness"],
                 knee_real=knee_real)

    # Sliding-window Theil-Sen plateau finder: informational cross-check only.
    # It is biased low on curved retarding branches and NEVER gates -- the
    # structural discriminator is knee_real. Skip it in batch_mode (~0.2 s/sweep).
    if not batch_mode:
        plat = theilsen_plateau_core(Vk, lnI, win_pts=win_pts, min_efold=efold_min)
        Te_plat = 1.0/plat["m_plateau"] if np.isfinite(plat["m_plateau"]) else np.nan
        plat_agree = (np.isfinite(Te_plat) and abs(Te_plat-bp["Te"])/bp["Te"] < 0.30)
        flags.update(Te_plateau=Te_plat, plateau_agree=bool(plat_agree),
                     plateau_ok=plat["ok"])

    # ---- tiered quality gate ------------------------------------------------
    # knee_real is the load-bearing test: does a two-line (retarding+saturation)
    # model beat a single line by a wide margin? When it fails, the "retarding"
    # slope is being set by a displaced/false knee (broad hump, no real
    # transition, or noise) and T_e is untrustworthy -> reject.
    # When it holds, T_e is reliable; r2 and efold only GRADE it, they don't veto.
    if not knee_real:
        flags["quality"] = "reject"; flags["pass"] = False
        flags["fail"] = "no_real_knee"
        return dict(Te=None, V_p=None, flags=flags)
    if bp["efold"] < efold_min:
        # too short a lever arm to trust the slope even with a real knee
        flags["quality"] = "reject"; flags["pass"] = False
        flags["fail"] = "short_lever_arm"
        return dict(Te=None, V_p=None, flags=flags)

    if r2 >= r2_high and bp["efold"] >= efold_high:
        flags["quality"] = "high"
    else:
        flags["quality"] = "medium"
    flags["pass"] = True

    # ---- two-population check ------------------------------------------------
    # The primary fit locks onto the SHARPEST bend, which in a two-temperature
    # sweep is the cold/hot corner -> it reports the COLD slope and puts V_p at
    # that corner. Re-run the same breakpoint fit on the branch ABOVE the knee:
    # if it splits into a steep hot segment + flat saturation, and the hot slope
    # is DISTINCTLY shallower (Te_hot/Te_cold >= hot_ratio) with enough lever
    # arm, report the HOT population instead and move V_p to the hot/sat corner.
    # On a clean single-population sweep the branch above the knee is flat, the
    # second fit fails its slope-ordering guard, and we keep the primary result.
    Te_cold  = bp["Te"]
    Vp_cold  = bp["V_p"]
    Vk_cold  = bp["V_knee"]
    m_use, b_use   = bp["m_ret"], bp["b_ret"]
    msat_use, bsat_use = bp["m_sat"], bp["b_sat"]
    Te_use, Vp_use, Vknee_use = Te_cold, Vp_cold, Vk_cold
    n_pop = 1
    ret_lo, ret_hi = Vk[0], Vk[bp.get("k_ret", bp["k"]) - 1]  # cold retarding bounds (endpoint-trimmed)
    sat_lo, sat_hi = Vk[bp["k"]], Vk[-1]

    if two_pop:
        split = two_population_split_core(
            Vk, lnI, bp["k"], min_seg=min_seg,
            sat_window_V=sat_window_V, robust_Te=True)
        if split["ok"]:
            hot_real = bool(split["sse"] < knee_ratio * split["sse_single"])
            ratio = split["Te_hot"] / Te_cold if Te_cold > 0 else np.inf
            distinct = (ratio >= hot_ratio)
            # A distinct hot population sits a modest factor above the bulk.
            # A very large ratio (default > 4x) is the rounded saturation
            # roll-off being misread as a second Maxwellian, not real physics.
            physical = (ratio <= hot_ratio_max)
            long_enough = (split["efold_hot"] >= hot_efold_min)
            flags.update(two_pop_found=True, Te_cold=Te_cold,
                         Te_hot=split["Te_hot"], hot_real=hot_real,
                         hot_efold=split["efold_hot"], hot_distinct=bool(distinct),
                         hot_ratio_val=float(ratio), hot_physical=bool(physical))
            if hot_real and distinct and physical and long_enough:
                # report the hotter population
                n_pop = 2
                Te_use   = split["Te_hot"]
                Vp_use   = split["V_p"]
                Vknee_use = split["V_hotsat"]
                m_use, b_use     = split["m_hot"], split["b_hot"]
                msat_use, bsat_use = split["m_sat"], split["b_sat"]
                ret_lo, ret_hi = Vk[bp["k"]], Vk[split["k2"]-1]   # hot segment
                sat_lo, sat_hi = Vk[split["k2"]], Vk[-1]
                # re-grade on the hot segment's own r2 / lever arm
                xk2 = Vk[bp["k"]:split["k2"]]; yk2 = lnI[bp["k"]:split["k2"]]
                yh2 = m_use*xk2 + b_use
                r2h = 1 - np.sum((yk2-yh2)**2)/np.sum((yk2-yk2.mean())**2) \
                      if yk2.size and np.sum((yk2-yk2.mean())**2)>0 else 0.0
                flags["r2_hot"] = r2h
                flags["quality"] = "high" if (r2h>=r2_high and
                    split["efold_hot"]>=efold_high) else "medium"
        else:
            flags["two_pop_found"] = False

    flags["n_populations"] = n_pop
    Te_final = Te_use if n_pop == 2 else bp["Te"]
    Vp_final = Vp_use if n_pop == 2 else bp["V_p"]
    Vkn_final = Vknee_use if n_pop == 2 else bp["V_knee"]

    # ---- plasma-potential refinement (single-population only) ---------------
    flags["vp_refined"] = False
    if refine_vp and n_pop == 1:
        rv = _refine_vp(Vk, lnI, bp["k"], m_use, b_use, min_seg=min_seg)
        if rv is not None:
            Vp_ref, ms_ref, bs_ref, sat_lo_ref = rv
            flags.update(vp_refined=True, V_p_corner=float(bp["V_p"]),
                         V_p_shift=float(Vp_ref - bp["V_p"]))
            Vp_final = Vp_ref
            msat_use, bsat_use = ms_ref, bs_ref
            sat_lo = sat_lo_ref

    # fit_region: everything a plotter needs to redraw the fit on the data
    fit_region = dict(
        V=Vk, lnI=lnI,                       # masked-log data actually fit
        retarding_V=(float(ret_lo), float(ret_hi)),
        saturation_V=(float(sat_lo), float(sat_hi)),
        m_ret=float(m_use), b_ret=float(b_use),
        esat_slope=float(msat_use), esat_intercept=float(bsat_use),
        n_populations=n_pop,
        old_V=(float(Vk[0]), float(Vk[bp.get("k_ret", bp["k"])-1])),
        cold_m_ret=float(bp["m_ret"]), cold_b_ret=float(bp["b_ret"]),
        cold_Te=float(Te_cold), cold_V_p=float(Vp_cold),
    )

    return dict(Te=Te_final*u.eV, V_p=Vp_final*u.V, V_knee=Vkn_final*u.V,
                m_ret=m_use, b_ret=b_use,
                esat_slope=msat_use, esat_intercept=bsat_use,
                floor_rms=floor_rms*u.A, fit_region=fit_region, flags=flags)
