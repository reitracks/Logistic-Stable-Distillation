import sys
sys.path.insert(1, '/Users/reiota/Desktop/conda/anaconda3/lib/python3.12/site-packages')
import numpy as np
import pandas as pd
from dataclasses import dataclass

EPS = np.nextafter(0.0, 1.0)

def simple_quantile_filter(u: float, u_prime: float, t: float):
    if u < t:
        return u, u_prime
    if u_prime > t:
        return u_prime, u
    u_tilde = t * ((u - t) / (1.0 - t))
    u_out = t + (1.0 - t) * (u_prime / t)
    #print("quantile filter activated")
    return u_out, u_tilde

def poibin_pmf(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    pmf = np.zeros(probs.size + 1, dtype=np.float64)
    pmf[0] = 1.0

    for j, p in enumerate(probs, start=1):
        prev = pmf[:j].copy()
        pmf[1:j + 1] = pmf[1:j + 1] * (1.0 - p) + prev * p
        pmf[0] *= (1.0 - p)

    return pmf

def randomized_residual_and_pvalue_fast(
    Sn: int,
    probs: np.ndarray,
    rng: np.random.Generator,
):
    pmf = poibin_pmf(probs)
    cdf = np.cumsum(pmf)

    omega = float(rng.random())
    G_Sn = 1.0 if Sn >= probs.size else float(cdf[Sn])
    G_prev = float(cdf[Sn - 1]) if Sn > 0 else 0.0

    W = float(np.clip(omega * G_Sn + (1.0 - omega) * G_prev, EPS, 1.0))
    side = 1 if W <= 0.5 else 0
    U = float(np.clip(2.0 * min(W, 1.0 - W), EPS, 1.0))
    return W, U, side, cdf

def poibin_ppf_from_cdf(cdf: np.ndarray, q: float) -> int:
    return int(np.searchsorted(cdf, q, side="left"))

def build_suffix_pmf(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    k = probs.size
    suff = np.zeros((k + 1, k + 1), dtype=np.float64)
    suff[k, 0] = 1.0

    for i in range(k - 1, -1, -1):
        p = probs[i]
        upto = k - i
        suff[i, 0] = suff[i + 1, 0] * (1.0 - p)
        suff[i, 1:upto + 1] = (
            suff[i + 1, 1:upto + 1] * (1.0 - p)
            + suff[i + 1, :upto] * p
        )

    return suff

def conditional_one_prob(
    s: int,
    r: int,
    probs: np.ndarray,
    suff: np.ndarray,
    i: int,
) -> float:
    t = r - s
    if t <= 0:
        return 0.0
    den = suff[i, t]
    if den <= 0.0:
        return 0.0
    return float(probs[i] * suff[i + 1, t - 1] / den)

@dataclass
class SDStepResult:
    U_j: float
    y_next: np.ndarray

def coupling_fast_active(
    probs: np.ndarray,
    y_active: np.ndarray,
    r: int,
    r_tilde: int,
    rng: np.random.Generator,
) -> np.ndarray:
    k = probs.size
    if k == 0:
        return y_active.copy()

    suff = build_suffix_pmf(probs)
    yt = np.empty(k, dtype=np.int8)

    s = 0
    s_tilde = 0

    for i in range(k - 1):
        kappa_og = conditional_one_prob(s, r, probs, suff, i)
        kappa_tilde = conditional_one_prob(s_tilde, r_tilde, probs, suff, i)

        p11 = min(kappa_og, kappa_tilde)
        p00 = 1.0 - max(kappa_og, kappa_tilde)
        p10 = max(kappa_og - kappa_tilde, 0.0)

        u = float(rng.random())

        if y_active[i]:
            marginal_p = p11 + p10
            yt[i] = 1 if u * marginal_p <= p11 else 0
        else:
            marginal_p = 1.0 - p11 - p10
            yt[i] = 0 if u * marginal_p <= p00 else 1

        s += int(y_active[i])
        s_tilde += int(yt[i])

    yt[k - 1] = 0 if s_tilde == r_tilde else 1
    return yt

def sd_one_step_fast(y, p_hat, x_col, t_filter, rng):
    y = np.asarray(y, dtype=np.int8)
    x = np.asarray(x_col, dtype=np.bool_)
    p_hat = np.asarray(p_hat, dtype=np.float64)

    # Important: for binary x, only the active positions matter.
    if not x.any():
        omega = float(rng.random())
        W = float(np.clip(omega, EPS, 1.0))
        side = 1 if W <= 0.5 else 0
        U = float(np.clip(2.0 * min(W, 1.0 - W), EPS, 1.0))
        U_prime = max(float(rng.random()), EPS)
        U_prime = float(np.clip(2.0 * min(U_prime, 1.0 - U_prime), EPS, 1.0))
        U_j, _ = simple_quantile_filter(U, U_prime, t_filter)
        return SDStepResult(U_j, y)

    probs = p_hat[x]      # selected Bernoulli probabilities
    y_active = y[x]       # selected binary outcomes
    Sn = int(y_active.sum())

    W, U, side, cdf = randomized_residual_and_pvalue_fast(Sn, probs, rng)

    U_prime = max(float(rng.random()), EPS)
    U_j, Ue = simple_quantile_filter(U, U_prime, t_filter)
    Wf = (Ue / 2.0) if side == 1 else (1.0 - Ue / 2.0)

    if W == Wf:
        return SDStepResult(U_j, y)

    r = poibin_ppf_from_cdf(cdf, W)
    r_tilde = poibin_ppf_from_cdf(cdf, Wf)

    if r == r_tilde:
        return SDStepResult(U_j, y)

    y_next = y.copy()
    y_next[x] = coupling_fast_active(probs, y_active, r, r_tilde, rng)
    return SDStepResult(U_j, y_next)

def run_sd_fast(
    feather_path,
    y_col,
    p_hat_col,
    x_prefix,
    p_cols,
    t_filter,
    seed,
    progress_every,
):
    df = pd.read_feather(feather_path)

    y = df[y_col].to_numpy(dtype=np.int8, copy=True)
    p_hat = df[p_hat_col].to_numpy(dtype=np.float64, copy=False)

    rng = np.random.default_rng(seed)
    U_out = np.empty(p_cols, dtype=np.float64)

    missing = [f"{x_prefix}{j}" for j in range(p_cols) if f"{x_prefix}{j}" not in df.columns]
    if missing:
        raise KeyError(f"Missing predictor columns, e.g. {missing[:3]}")

    for j in range(p_cols):
        x_col = df[f"{x_prefix}{j}"].to_numpy(dtype=np.bool_, copy=False)

        res = sd_one_step_fast(y, p_hat, x_col, t_filter, rng)
        U_out[j] = res.U_j
        y = res.y_next

        if progress_every and (j + 1) % progress_every == 0:
            print(f"Processed {j + 1}/{p_cols}.")

    return U_out

if __name__ == "__main__":
    U_vals = run_sd_fast(
        "binary_real_30000_v4.feather",
        "y",
        "p_hat",
        "x",
        30000,
        0.0005,
        123,
        1000,
    )
    np.save('binary_u_val_test_30000_v4', U_vals)