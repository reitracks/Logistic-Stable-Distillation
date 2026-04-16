import sys
sys.path.insert(1, '/Users/reiota/Desktop/conda/anaconda3/lib/python3.12/site-packages')
import numpy as np
import pandas as pd
from scipy.stats import cauchy

EPS = np.nextafter(0.0, 1.0)

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

    #print([len(cdf), Sn, G_Sn, Sn/len(cdf), np.mean(probs)])

    if (Sn >= probs.size):
        print("clip")

    W = float(np.clip(omega * G_Sn + (1.0 - omega) * G_prev, EPS, 1.0))
    side = 1 if W <= 0.5 else 0
    U = float(np.clip(2.0 * min(W, 1.0 - W), EPS, 1.0))
    return W, U, side, cdf

def poibin_ppf_from_cdf(cdf: np.ndarray, q: float) -> int:
    return int(np.searchsorted(cdf, q, side="left"))

def sd_one_step_fast(y, p_hat, x_col, t_filter, rng):
    y = np.asarray(y, dtype=np.int8)
    x = np.asarray(x_col, dtype=np.bool_)
    p_hat = np.asarray(p_hat, dtype=np.float64)

    probs = p_hat[x]      # selected Bernoulli probabilities
    y_active = y[x]       # selected binary outcomes
    Sn = int(y_active.sum())

    W, U, side, cdf = randomized_residual_and_pvalue_fast(Sn, probs, rng)
    #return np.random.random()
    return U

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
    p_vals = [0]*p_cols

    y = df[y_col].to_numpy(dtype=np.int8, copy=True)
    p_hat = df[p_hat_col].to_numpy(dtype=np.float64, copy=False)
    rng = np.random.default_rng(seed)

    missing = [f"{x_prefix}{j}" for j in range(p_cols) if f"{x_prefix}{j}" not in df.columns]
    if missing:
        raise KeyError(f"Missing predictor columns, e.g. {missing[:3]}")

    for j in range(p_cols):
        x_col = df[f"{x_prefix}{j}"].to_numpy(dtype=np.bool_, copy=False)

        res = sd_one_step_fast(y, p_hat, x_col, t_filter, rng)
        p_vals[j] = res

        if progress_every and (j + 1) % progress_every == 0:
            print(f"Processed {j + 1}/{p_cols}.")

    return p_vals

def cauchy_combination_test(pvalues, weights=None):
    """
    Performs the Cauchy Combination Test (CCT).
    
    Args:
        pvalues: list or numpy array of p-values.
        weights: list or numpy array of weights (optional).
        
    Returns:
        combined_p_value
    """
    pvalues = np.array(pvalues)
    if weights is None:
        weights = np.ones(len(pvalues)) / len(pvalues)
    else:
        weights = np.array(weights)
        weights = weights / np.sum(weights)

    # Cauchy transformation: tan((0.5 - p) * pi)
    # Using scipy.stats.cauchy.isf (Inverse Survival Function)
    # is equivalent to tan((0.5 - p) * pi) for standard Cauchy
    test_statistic = np.sum(weights * cauchy.isf(pvalues))
    
    # Combined P-value calculation
    # P = 1 - cauchy.cdf(test_statistic)
    # Since Cauchy is symmetric, this can be simplified
    combined_p_value = 0.5 - np.arctan(test_statistic) / np.pi
    
    return combined_p_value

if __name__ == "__main__":
    p_vals = run_sd_fast(
        "binary_20k_v2_6.feather",
        "y",
        "p_hat",
        "x",
        60000,
        1/60000,
        123,
        10000,
    )
    p_vals.sort()
    combined_p_val = cauchy_combination_test(p_vals)
    print(f"Cauchy combinration test result: {combined_p_val}")
    print(f"Bonferoni corrected p value: {min(1.0, p_vals[0]*len(p_vals))}")
    