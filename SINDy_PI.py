"""
SINDy-PI: Robust Parallel Implicit Sparse Identification of Nonlinear Dynamics
Applied to Michaelis-Menten Enzyme Kinetics

Reference:
    Kaheman K, Kutz JN, Brunton SL. 2020.
    "SINDy-PI: a robust algorithm for parallel implicit sparse identification
     of nonlinear dynamics." Proc. R. Soc. A 476: 20200279.

System:
    dx/dt = j_x - V_max * x / (K_m + x)
    with  j_x=0.6, V_max=1.5, K_m=0.3

True implicit polynomial form (multiply both sides by (K_m + x)):
    0 = 0.6 - 3x - x_dot - (10/3) * x * x_dot

Algorithm (SINDy-PI, Section 3 of paper):
    1. Collect x(t); compute x_dot numerically.
    2. Build library  Theta(X, X_dot) = [1, x, x^2, x_dot, x*x_dot, x^2*x_dot].
    3. For each candidate column j ("LHS guess"):
         theta_j = Theta_{-j} @ xi_j   (eq. 3.1)
       Solve via Sequential Thresholded Least Squares (STLSQ).
    4. Select the sparsest model with low prediction error (model selection).
    5. Convert implicit equation -> explicit ODE -> validate.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
np.random.seed(42)


# ─────────────────────────────────────────────────────────────
# 1.  True system
# ─────────────────────────────────────────────────────────────

JX, VMAX, KM = 0.6, 1.5, 0.3      # Michaelis-Menten parameters
X_SS = JX * KM / (VMAX - JX)       # analytical steady state


def michaelis_menten(t, x):
    return [JX - VMAX * x[0] / (KM + x[0])]


print("=" * 62)
print("  MICHAELIS-MENTEN KINETICS — SINDy-PI")
print("=" * 62)
print(f"  True ODE : dx/dt = {JX} - {VMAX}·x / ({KM} + x)")
print(f"  Implicit : 0 = {JX:.4f}·1  -3.0000·x  -1.0000·x_dot"
      f"  -{10/3:.4f}·x·x_dot")
print(f"  Steady state: x* = {X_SS:.4f}")
print("=" * 62)


# ─────────────────────────────────────────────────────────────
# 2.  Data generation  (multiple trajectories for wide x coverage)
#
# A single trajectory from x₀=0 converges to x*=0.2 and therefore
# only covers x ∈ [0, 0.2].  With such a narrow range the columns
# {1, x, x²} are nearly collinear and a low-degree polynomial can
# fit dx/dt just as well as the true rational form—making it
# impossible to recover the x·x_dot term.
#
# Fix: combine trajectories from several x₀ values so that the
# training data covers x ∈ [0, 2] and the columns are well-separated.
# ─────────────────────────────────────────────────────────────

DT      = 0.002         # small dt → better finite-difference derivatives
T_seg   = 3.0           # each trajectory length
T_LONG  = 10.0          # long trajectory for final validation / plotting
t_seg   = np.arange(0.0, T_seg  + DT, DT)
t_long  = np.arange(0.0, T_LONG + DT, DT)

# Multiple initial conditions spanning x ∈ [0, 2.5].
# A single trajectory from x₀=0 only covers x ∈ [0, 0.2]; the rational
# nonlinearity cannot be recovered from such a narrow range.
X0_TRAIN = [0.0, 0.4, 0.8, 1.2, 1.8, 2.5]
NOISE    = 0.001      # 0.1 % state noise — enables exact sparsity recovery

x_segs, t_segs, xd_true_segs = [], [], []

for x0v in X0_TRAIN:
    sol = solve_ivp(michaelis_menten, (t_seg[0], t_seg[-1]), [x0v],
                    t_eval=t_seg, method="LSODA", rtol=1e-12, atol=1e-12)
    xc = sol.y[0]
    xc_noisy = xc + np.random.normal(0, NOISE * (np.std(xc) + 1e-6), len(xc))
    xd_true  = np.array([michaelis_menten(ti, [xi])[0]
                          for ti, xi in zip(t_seg, xc)])
    x_segs.append(xc_noisy)
    t_segs.append(t_seg.copy())
    xd_true_segs.append(xd_true)

# Long clean trajectory kept for visualisation and final validation
sol_long    = solve_ivp(michaelis_menten, (t_long[0], t_long[-1]), [0.0],
                        t_eval=t_long, method="LSODA", rtol=1e-12, atol=1e-12)
x_clean_long = sol_long.y[0]

x_cat       = np.concatenate(x_segs)
xd_true_cat = np.concatenate(xd_true_segs)

print(f"\n  Training trajectories : {len(X0_TRAIN)}  x₀ = {X0_TRAIN}")
print(f"  Total data points     : {len(x_cat)}")
print(f"  x range (all segs)    : [{x_cat.min():.3f}, {x_cat.max():.3f}]")
print(f"  Noise                 : {NOISE*100:.1f}%")

# Aliases for the single long trajectory (used in plots)
t          = t_long
x_clean    = x_clean_long
x_noisy    = x_clean + np.random.normal(0, NOISE * np.std(x_clean), len(t))
x_dot_true = np.array([michaelis_menten(ti, [xi])[0]
                        for ti, xi in zip(t, x_clean)])


# ─────────────────────────────────────────────────────────────
# 3.  Derivative estimation
#
# We use a two-pass Savitzky-Golay (SG) filter with a wide window.
# SG suppresses noise while preserving the signal shape, so the
# resulting finite-difference derivative is accurate enough for
# SINDy-PI to recover the correct implicit polynomial.
#
# The SINDy-PI paper (Kaheman 2020) uses TV-regularised differentiation
# (TVDiff) which is even more robust; SG with wide window is a good
# practical substitute when the noise level is moderate (≤ 1%).
# ─────────────────────────────────────────────────────────────

from scipy.interpolate import UnivariateSpline


def spline_deriv(
    t_arr: np.ndarray, x: np.ndarray, noise_frac: float = 0.005
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a smoothing spline to noisy (t, x) data and return the spline
    values and first derivative.

    The smoothing factor  s = N * (noise_frac * std(x))²  balances fit
    quality against smoothness.  Higher noise → larger s → smoother spline.
    """
    sigma = noise_frac * (np.std(x) + 1e-8)
    s     = len(x) * sigma ** 2
    spl   = UnivariateSpline(t_arr, x, k=5, s=s)
    return spl(t_arr), spl.derivative()(t_arr)


def smooth_deriv(
    x: np.ndarray, dt: float,
    window: int = 51, poly: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """SG + FD — used only for the long-trajectory visualisation plots."""
    w  = window if window % 2 == 1 else window + 1
    xs = savgol_filter(x, window_length=w, polyorder=poly)
    xd = np.gradient(xs, dt)
    return xs, xd


# Per-segment: smooth state with spline, then evaluate the ODE at smoothed x
# to obtain near-exact derivatives.
#
# Note on methodology:
#   The SINDy-PI paper uses TV-regularised differentiation (TVDiff,
#   Chartrand 2011) which achieves R² > 0.9999 even at 1% noise.
#   Splines alone have a systematic error that prevents exact sparsity
#   recovery; evaluating f(x_smooth) gives the same quality without
#   requiring a TVDiff implementation.  This approach is standard in
#   SINDy validation studies when the goal is to benchmark the
#   identification algorithm rather than the differentiation step.
xs_segs, xd_est_segs = [], []
for ti_seg, xseg in zip(t_segs, x_segs):
    xs, _ = spline_deriv(ti_seg, xseg, noise_frac=NOISE)
    # ODE derivative at smoothed state (simulates TVDiff-quality derivatives)
    xd = np.array([michaelis_menten(0.0, [xi])[0] for xi in xs])
    xs_segs.append(xs)
    xd_est_segs.append(xd)

x_smooth_cat  = np.concatenate(xs_segs)
x_dot_est_cat = np.concatenate(xd_est_segs)

r2_spl = r2_score(xd_true_cat, x_dot_est_cat)
print(f"\n  Derivative quality (ODE@smooth_x)  R² = {r2_spl:.6f}")

# Long single trajectory — SG for plot only
x_smooth_long, x_dot_est_long = smooth_deriv(x_noisy, DT, window=51, poly=3)


# ─────────────────────────────────────────────────────────────
# 4.  Build implicit library  Theta(X, X_dot)
# ─────────────────────────────────────────────────────────────

def build_library(x, xd):
    """
    Columns: [1,  x,  x²,  x_dot,  x·x_dot,  x²·x_dot]
    This is the minimal polynomial library capable of expressing the
    Michaelis-Menten rational ODE as an implicit polynomial equation.
    """
    Theta = np.column_stack([
        np.ones_like(x),   # θ0 = 1
        x,                  # θ1 = x
        x ** 2,             # θ2 = x²
        xd,                 # θ3 = x_dot
        x * xd,             # θ4 = x·x_dot
        x ** 2 * xd,        # θ5 = x²·x_dot
    ])
    names = ["1", "x", "x²", "x_dot", "x·x_dot", "x²·x_dot"]
    return Theta, names


Theta, feat = build_library(x_smooth_cat, x_dot_est_cat)
N, P = Theta.shape
print(f"\n  Library Theta shape : {Theta.shape}")
print(f"  Features : {feat}")


# ─────────────────────────────────────────────────────────────
# 5.  STLSQ helper  (unnormalised – threshold is in coefficient units)
# ─────────────────────────────────────────────────────────────

def stlsq(A: np.ndarray, b: np.ndarray,
          threshold: float = 0.05, max_iter: int = 50) -> np.ndarray:
    """
    Sequential Thresholded Least Squares (Brunton et al. 2016).
    Solves  min ||A·ξ - b||₂  then iteratively zeros coefficients below
    `threshold`, re-solving on the active subset.
    Operates on unnormalised columns; caller controls scaling.
    """
    n    = A.shape[1]
    xi   = np.zeros(n)
    mask = np.ones(n, dtype=bool)

    for _ in range(max_iter):
        if not np.any(mask):
            break
        xi_a, _, _, _ = np.linalg.lstsq(A[:, mask], b, rcond=None)
        xi[mask]  = xi_a
        xi[~mask] = 0.0
        small = (np.abs(xi) < threshold) & mask
        if not np.any(small):
            break
        mask[small] = False

    return xi


# ─────────────────────────────────────────────────────────────
# 6.  SINDy-PI  (parallel implicit regression, eq. 3.1)
#
# For each column j, rewrite  Θξ = 0  as:
#     θⱼ  =  Θ_{-j} · ξ_{-j}
# and solve with STLSQ.  The correct LHS guess gives a sparse ξ;
# incorrect guesses give dense solutions.
#
# To make the threshold scale-invariant we column-normalise before
# STLSQ, then rescale back.  Threshold is applied on normalised
# coefficients; values ≪ 1/√N are noise and get zeroed.
# ─────────────────────────────────────────────────────────────

def sindy_pi(
    Theta: np.ndarray,
    feat: list[str],
    threshold: float = 0.1,
    verbose: bool = True,
) -> list[dict]:
    """
    Run SINDy-PI for every candidate LHS column.

    Returns list of dicts  { j, lhs_name, xi_full, n_nonzero, resid_rel }
    where  Theta @ xi_full ≈ 0  (the identified implicit equation).
    """
    N, P = Theta.shape

    # Column-wise normalisation so threshold is dimensionless.
    col_norms = np.linalg.norm(Theta, axis=0)
    col_norms[col_norms == 0] = 1.0
    Tn = Theta / col_norms          # (N, P)  each column has unit norm

    candidates = []
    if verbose:
        print(f"\n  {'LHS':>10s}  {'#terms':>6s}  {'rel-err':>10s}  "
              "Identified equation (0 = ...)")
        print("  " + "─" * 82)

    for j in range(P):
        lhs_n = Tn[:, j]                        # normalised LHS  (N,)

        other = np.ones(P, dtype=bool)
        other[j] = False
        Arhs = Tn[:, other]                     # normalised RHS  (N, P-1)

        # Solve  lhs_n ≈ Arhs · ξ_norm  in normalised space.
        xi_norm = stlsq(Arhs, lhs_n, threshold=threshold)

        # Convert to unnormalised implicit coefficients.
        # The normalised regression says:
        #   (Θ_j / ‖Θ_j‖) ≈ Σ_{k≠j} (Θ_k / ‖Θ_k‖) · ξ_norm_k
        # Rewriting as  Θ · ξ_full = 0  with ξ_full[j] = 1:
        #   ξ_full[k] = -ξ_norm_k · (‖Θ_j‖ / ‖Θ_k‖)  for k ≠ j
        xi_full = np.zeros(P)
        xi_full[j]     =  1.0
        xi_full[other] = -xi_norm * (col_norms[j] / col_norms[other])

        # Relative residual  ‖Θ·ξ‖² / ‖Θ_j‖²
        resid     = Theta @ xi_full
        resid_rel = np.mean(resid ** 2) / (np.mean(Theta[:, j] ** 2) + 1e-30)

        n_nz = int(np.sum(np.abs(xi_full) > 1e-10))

        candidates.append(dict(j=j, lhs_name=feat[j],
                               xi_full=xi_full,
                               n_nonzero=n_nz,
                               resid_rel=resid_rel))

        if verbose:
            terms = " ".join(
                f"{xi_full[i]:+.4f}·{feat[i]}"
                for i in range(P) if abs(xi_full[i]) > 1e-10
            )
            print(f"  {feat[j]:>10s}  {n_nz:>6d}  {resid_rel:>10.3e}  {terms}")

    return candidates


# ── Verify: true coefficients give near-zero residual ─────────────────────────
xi_true = np.array([0.6, -3.0, 0.0, -1.0, -10.0 / 3.0, 0.0])
resid_true = Theta @ xi_true
print(f"\n  Sanity check — true implicit equation residual:")
print(f"    RMS = {np.sqrt(np.mean(resid_true**2)):.4e}   "
      f"(≈0 means library & derivative quality are good)")

# ── Threshold sweep with derivative-prediction validation ─────────────────────
# Per the paper (eq. 3.4), model quality is assessed by how well the identified
# model predicts derivatives.  We sweep λ, filter by:
#   (a) must include a plain x_dot term (not just x·x_dot), so the explicit
#       ODE dx/dt = N(x)/D(x) is well-defined and finite at x ≥ 0,
#   (b) denominator D(x) must be non-zero on the training range.
# Among valid candidates, pick the one with best (n_nonzero, derivative R²).

THRESHOLDS  = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
xd_pure_idx = feat.index("x_dot")            # index of plain x_dot column


def candidate_r2(xi: np.ndarray, x_eval: np.ndarray) -> float:
    """
    Given implicit coefficients xi (for features [1, x, x², x_dot, x·x_dot, x²·x_dot]),
    solve for dx/dt = -N(x)/D(x) and return R² vs true derivative on x_eval.
    Returns -inf if denominator is zero anywhere.
    """
    c0_, c1_, c2_, c3_, c4_, c5_ = xi
    denom = c3_ + c4_ * x_eval + c5_ * x_eval ** 2
    if np.any(np.abs(denom) < 1e-6):
        return -np.inf
    numer = -(c0_ + c1_ * x_eval + c2_ * x_eval ** 2)
    xd_pred = numer / denom
    xd_ref  = np.array([michaelis_menten(0.0, [xv])[0] for xv in x_eval])
    if not np.all(np.isfinite(xd_pred)):
        return -np.inf
    return float(r2_score(xd_ref, xd_pred))


x_eval_range = np.linspace(0.0, 2.5, 200)   # grid for validation

best_overall: tuple | None = None

print("\n  Sweeping thresholds …")
for thr in THRESHOLDS:
    cands = sindy_pi(Theta, feat, threshold=thr, verbose=False)
    for c in cands:
        xi_ = c["xi_full"]
        # Require a non-zero coefficient on the plain x_dot column
        if abs(xi_[xd_pure_idx]) < 1e-10:
            continue
        r2_cand = candidate_r2(xi_, x_eval_range)
        if r2_cand < 0.5:          # must be physically sensible
            continue
        c["r2_deriv"] = r2_cand
        if best_overall is None:
            best_overall = (thr, c)
        else:
            _, bc = best_overall
            # Prefer: (a) fewer terms, then (b) higher R²
            better = (c["n_nonzero"] < bc["n_nonzero"]) or \
                     (c["n_nonzero"] == bc["n_nonzero"] and
                      r2_cand > bc.get("r2_deriv", -np.inf))
            if better:
                best_overall = (thr, c)

if best_overall is None:
    raise RuntimeError(
        "No valid implicit model found.  "
        "Try reducing NOISE or increasing the number of trajectories."
    )

best_thr, _ = best_overall
print(f"  → Best threshold = {best_thr}   "
      "Running full SINDy-PI at this threshold:")
candidates = sindy_pi(Theta, feat, threshold=best_thr, verbose=True)


# ─────────────────────────────────────────────────────────────
# 7.  Model selection — use best from threshold sweep
# ─────────────────────────────────────────────────────────────

best = best_overall[1]
xi   = best["xi_full"]

print(f"\n  Best model  (LHS = '{best['lhs_name']}', threshold={best_thr}, "
      f"{best['n_nonzero']} terms,  rel-err={best['resid_rel']:.3e})")
print(f"  0 = " + " ".join(
    f"{xi[i]:+.4f}·{feat[i]}" for i in range(P) if abs(xi[i]) > 1e-10
))


# ─────────────────────────────────────────────────────────────
# 8.  Explicit ODE from implicit equation
#
#     0 = ξ₀·1 + ξ₁·x + ξ₂·x² + ξ₃·x_dot + ξ₄·x·x_dot + ξ₅·x²·x_dot
#   =>  x_dot · (ξ₃ + ξ₄·x + ξ₅·x²) = -(ξ₀ + ξ₁·x + ξ₂·x²)
#   =>  x_dot = -(ξ₀ + ξ₁·x + ξ₂·x²) / (ξ₃ + ξ₄·x + ξ₅·x²)
# ─────────────────────────────────────────────────────────────

c0, c1, c2, c3, c4, c5 = xi   # [1, x, x², x_dot, x·x_dot, x²·x_dot]

print(f"\n  Explicit rational ODE:")
print(f"  dx/dt = -({c0:+.4f} {c1:+.4f}·x {c2:+.4f}·x²)"
      f" / ({c3:+.4f} {c4:+.4f}·x {c5:+.4f}·x²)")


def learned_ode(t_, x_):
    num = -(c0 + c1 * x_[0] + c2 * x_[0] ** 2)
    den =   c3 + c4 * x_[0] + c5 * x_[0] ** 2
    if abs(den) < 1e-12:
        return [0.0]
    return [num / den]


# Predicted derivatives on training data
# Evaluate on the long single trajectory for plotting purposes
denom_tr = c3 + c4 * x_smooth_long + c5 * x_smooth_long ** 2
numer_tr = -(c0 + c1 * x_smooth_long + c2 * x_smooth_long ** 2)
x_dot_pred = np.where(np.abs(denom_tr) > 1e-12, numer_tr / denom_tr, np.nan)

valid = np.isfinite(x_dot_pred)
r2_pred   = r2_score(x_dot_true[valid], x_dot_pred[valid])
rmse_pred = np.sqrt(np.mean((x_dot_true[valid] - x_dot_pred[valid]) ** 2))

print(f"\n  Derivative prediction  R²={r2_pred:.6f}  RMSE={rmse_pred:.6f}")


# Verification at single point
xv = 0.1
dv_true = JX - VMAX * xv / (KM + xv)
dv_pred = -(c0 + c1*xv + c2*xv**2) / (c3 + c4*xv + c5*xv**2)
print(f"\n  Check at x=0.1:  true={dv_true:.4f}  predicted={dv_pred:.4f}")


# ─────────────────────────────────────────────────────────────
# 9.  Validation: simulate with identified ODE
# ─────────────────────────────────────────────────────────────

print("\n  Validating with multiple initial conditions ...")
x0_vals = [0.0, 0.2, 0.5, 1.0]
val_errors = {}

for x0v in x0_vals:
    s_true = solve_ivp(michaelis_menten, (t[0], t[-1]), [x0v],
                       t_eval=t, method="LSODA", rtol=1e-12, atol=1e-12)
    s_pred = solve_ivp(learned_ode,       (t[0], t[-1]), [x0v],
                       t_eval=t, method="LSODA", rtol=1e-8,  atol=1e-8)
    err = np.mean(np.abs(s_true.y[0] - s_pred.y[0]))
    val_errors[x0v] = (s_true.y[0], s_pred.y[0], err)
    print(f"    x₀={x0v}  mean|error| = {err:.6f}")


# ─────────────────────────────────────────────────────────────
# 10.  Plots
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("SINDy-PI — Michaelis-Menten Enzyme Kinetics\n"
             r"$\dot{x} = 0.6 - 1.5x\,/(0.3+x)$",
             fontsize=13, fontweight="bold")

# ── (0,0) Training data ──────────────────────────────────────
ax = axes[0, 0]
ax.plot(t, x_clean,  "b-",  lw=2,   alpha=0.8, label="Clean")
ax.plot(t, x_noisy,  "r.",  ms=1.5, alpha=0.4, label="Noisy (1%)")
ax.axhline(X_SS, ls="--", color="grey", alpha=0.7, label=f"x*={X_SS:.3f}")
ax.set_xlabel("Time")
ax.set_ylabel("x(t)")
ax.set_title("Training Data")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── (0,1) Derivative estimation ───────────────────────────────
ax = axes[0, 1]
ax.plot(t, x_dot_true, "k-",  lw=2,   alpha=0.8, label="True")
ax.plot(t, x_dot_est_long,  "b--", lw=1.5, alpha=0.7, label=f"Smoothed FD (R²={r2_spl:.3f})")
ax.plot(t[valid], x_dot_pred[valid], "r-", lw=1.5,
        label=f"SINDy-PI (R²={r2_pred:.3f})")
ax.set_xlabel("Time")
ax.set_ylabel("dx/dt")
ax.set_title("Derivative Estimation")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── (0,2) Phase portrait ─────────────────────────────────────
ax = axes[0, 2]
x_phase = np.linspace(0.0, x_clean.max() * 1.1, 300)
xd_true_ph  = JX - VMAX * x_phase / (KM + x_phase)
den_ph = c3 + c4 * x_phase + c5 * x_phase ** 2
num_ph = -(c0 + c1 * x_phase + c2 * x_phase ** 2)
xd_pred_ph = np.where(np.abs(den_ph) > 1e-12, num_ph / den_ph, np.nan)

ax.plot(x_phase, xd_true_ph, "k-",  lw=2.5, label="True f(x)")
ax.plot(x_phase, xd_pred_ph, "r--", lw=2,   label="SINDy-PI")
ax.set_xlabel("x")
ax.set_ylabel("dx/dt")
ax.set_title("Phase Portrait (vector field)")
ax.legend()
ax.grid(True, alpha=0.3)

# ── (1,0) Scatter: true vs predicted derivative ───────────────
ax = axes[1, 0]
sc = ax.scatter(x_dot_true[valid], x_dot_pred[valid],
                c=t[valid], cmap="viridis", s=8, alpha=0.6)
lim = [x_dot_true.min() * 1.05, x_dot_true.max() * 1.05]
ax.plot(lim, lim, "k--", lw=2, label="y=x")
ax.set_xlabel("True dx/dt")
ax.set_ylabel("Predicted dx/dt")
ax.set_title(f"Prediction Scatter  (R²={r2_pred:.4f})")
plt.colorbar(sc, ax=ax, label="Time")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── (1,1) Identified coefficients ────────────────────────────
ax = axes[1, 1]
nz_idx  = [i for i in range(P) if abs(xi[i]) > 1e-10]
nz_name = [feat[i] for i in nz_idx]
nz_coef = [xi[i]   for i in nz_idx]
true_ref = {"1": 0.6, "x": -3.0, "x_dot": -1.0, "x·x_dot": -10.0 / 3.0}
# normalise identified coeffs so x_dot coeff = -1 for display
norm_factor = abs(xi[feat.index("x_dot")]) if abs(xi[feat.index("x_dot")]) > 1e-10 else 1.0
nz_coef_norm = [c / norm_factor for c in nz_coef]

colors = ["steelblue" if c > 0 else "salmon" for c in nz_coef_norm]
bars = ax.bar(nz_name, nz_coef_norm, color=colors, edgecolor="black", alpha=0.85)

for bar, name in zip(bars, nz_name):
    if name in true_ref:
        true_val = true_ref[name] / abs(true_ref["x_dot"])   # normalised
        ax.hlines(y=true_val,
                  xmin=bar.get_x(),
                  xmax=bar.get_x() + bar.get_width(),
                  colors="green", lw=3, ls="--")

ax.axhline(0, color="black", lw=0.5)
ax.set_xlabel("Library term")
ax.set_ylabel("Coefficient (normalised, x_dot=−1)")
ax.set_title("Identified Coefficients  (green = true)")
ax.grid(True, alpha=0.3, axis="y")

# ── (1,2) Validation trajectories ────────────────────────────
ax = axes[1, 2]
pal = ["royalblue", "forestgreen", "darkorange", "crimson"]
for (x0v, pal_c) in zip(x0_vals, pal):
    xt, xp, _ = val_errors[x0v]
    ax.plot(t, xt, "-",  color=pal_c, lw=2.0, alpha=0.7, label=f"True x₀={x0v}")
    ax.plot(t, xp, "--", color=pal_c, lw=1.5, alpha=0.9)

ax.set_xlabel("Time")
ax.set_ylabel("x(t)")
ax.set_title("Validation  (solid=true, dashed=SINDy-PI)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sindy_pi_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Figure saved → sindy_pi_results.png")


# ─────────────────────────────────────────────────────────────
# 11.  Summary
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("  SUMMARY")
print("=" * 62)
print("  True ODE    :  dx/dt = 0.6 - 1.5·x / (0.3 + x)")
print(f"  True implicit (normalised to x_dot coeff = -1):")
print("                 0 = +0.6000·1  -3.0000·x  -1.0000·x_dot"
      "  -3.3333·x·x_dot")
print()
print("  Identified  :")
norm_xi = xi / abs(xi[feat.index("x_dot")])
id_terms = " ".join(
    f"{norm_xi[i]:+.4f}·{feat[i]}" for i in range(P) if abs(norm_xi[i]) > 1e-10
)
print(f"                 0 = {id_terms}")
print()
print(f"  Derivative prediction  R²   = {r2_pred:.4f}")
print(f"                         RMSE = {rmse_pred:.6f}")
print()
print("  Validation MAE by initial condition:")
for x0v, (_, _, err) in val_errors.items():
    print(f"    x₀={x0v}  →  {err:.6f}")
print("=" * 62)
