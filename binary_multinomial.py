import sys
sys.path.insert(1, '/Users/reiota/Desktop/conda/anaconda3/lib/python3.12/site-packages')
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List

EPS = np.nextafter(0.0, 1.0)

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def simple_quantile_filter(u: float, u_prime: float, t: float) -> Tuple[float, float]:
    if u < t:
        return u, u_prime
    if u_prime > t:
        return u_prime, u
    u_tilde = t * ((u - t) / (1.0 - t))
    u_out = t + (1.0 - t) * (u_prime / t)
    return u_out, u_tilde

def y_matrix_to_K(y_mat: np.ndarray) -> np.ndarray:
    """
    Convert monotone binary rows:
      K = 0 -> (0,0,0,0,0,0,0,0,0,0)
      K = 1 -> (1,1,1,1,1,1,1,1,1,1)
      K = 2 -> (0,1,1,1,1,1,1,1,1,1)
      ...
      K = 10 -> (0,0,0,0,0,0,0,0,0,1)
    """
    y_mat = np.asarray(y_mat, dtype=np.int8)

    if y_mat.ndim != 2 or y_mat.shape[1] != 10:
        raise ValueError("y_mat must have shape (n_samples, 10).")

    if np.any((y_mat != 0) & (y_mat != 1)):
        raise ValueError("All y values must be binary 0/1.")

    # Must be monotone nondecreasing across the 10 coordinates.
    if np.any(y_mat[:, 1:] < y_mat[:, :-1]):
        raise ValueError("Each response row must be monotone: 0...0 1...1.")

    has_one = y_mat.any(axis=1)
    first_one = np.argmax(y_mat == 1, axis=1) + 1
    K = np.where(has_one, first_one, 0).astype(np.int16)
    return K

def K_to_y_matrix(K: np.ndarray, response_dim: int = 10) -> np.ndarray:
    """
    Inverse mapping back to y1,...,y10 if needed later.
    """
    K = np.asarray(K, dtype=np.int16)
    levels = np.arange(1, response_dim + 1, dtype=np.int16)
    y_mat = ((K[:, None] > 0) & (K[:, None] <= levels[None, :])).astype(np.int8)
    return y_mat

def logits_to_K_pmf(p_hat_logits: np.ndarray) -> np.ndarray:
    """
    p_hat_logits has shape (n_samples, 10).
    Each column t corresponds to the logit of:
        P(y_t = 0 | y_{t-1} = 0)
    with the convention that for t=1 this means P(y_1 = 0).

    Let q_t = P(still zero at time t | still zero up to time t-1).

    Then:
        P(K=1)  = 1 - q_1
        P(K=2)  = q_1 (1 - q_2)
        ...
        P(K=10) = q_1 ... q_9 (1 - q_10)
        P(K=0)  = q_1 ... q_10
    """
    p_hat_logits = np.asarray(p_hat_logits, dtype=np.float64)

    if p_hat_logits.ndim != 2 or p_hat_logits.shape[1] != 10:
        raise ValueError("p_hat_logits must have shape (n_samples, 10).")

    q = sigmoid(p_hat_logits)  # convert logits to probabilities
    n = q.shape[0]

    pmf = np.empty((n, 11), dtype=np.float64)
    survival = np.ones(n, dtype=np.float64)

    for k in range(1, 11):
        pmf[:, k] = survival * (1.0 - q[:, k - 1])
        survival *= q[:, k - 1]

    pmf[:, 0] = survival

    # Clean up any tiny numerical drift.
    pmf = np.clip(pmf, 0.0, 1.0)
    pmf /= pmf.sum(axis=1, keepdims=True)
    return pmf


def randomized_residual_and_pvalue(Sn: int, pmf: np.ndarray, rng: np.random.Generator):
    """
    Generic version for an arbitrary discrete distribution of Sn.
    """
    pmf = np.asarray(pmf, dtype=np.float64)
    cdf = np.cumsum(pmf)
    cdf[-1] = 1.0

    omega = float(rng.uniform(0.0, 1.0))
    G_Sn = float(cdf[Sn]) if 0 <= Sn < len(cdf) else 1.0
    G_prev = float(cdf[Sn - 1]) if Sn - 1 >= 0 else 0.0

    W = omega * G_Sn + (1.0 - omega) * G_prev
    W = float(np.clip(W, EPS, 1.0))

    side = 1 if W <= 0.5 else 0
    U = 2.0 * min(W, 1.0 - W)
    U = float(np.clip(U, EPS, 1.0))

    return W, U, side, cdf


def ppf_from_cdf(q: float, cdf: np.ndarray) -> int:
    return int(np.searchsorted(cdf, q, side="left"))


def build_suffix_pmfs(K_pmf: np.ndarray, x: np.ndarray) -> List[np.ndarray]:
    """
    suffix[i] is the PMF of:
        sum_{l=i}^{n-1} K_l X_l
    suffix[n] is delta_0.
    """
    x = np.asarray(x, dtype=np.uint8)
    n = x.size

    suffix = [None] * (n + 1)
    suffix[n] = np.array([1.0], dtype=np.float64)

    for i in range(n - 1, -1, -1):
        if x[i] == 0:
            suffix[i] = suffix[i + 1]
        else:
            suffix[i] = np.convolve(K_pmf[i], suffix[i + 1])

    return suffix

def kappa(s: int, r: int, K_pmf_i: np.ndarray, suffix_i: np.ndarray, suffix_ip1: np.ndarray, x_i: int) -> np.ndarray:
    """
    Returns the vector:
        kappa_mu(s, r) = p_mu * q_i(r - s - mu * x_i) / q_{i-1}(r - s)
    for mu = 0,...,10.
    """
    t = r - s
    out = np.zeros(11, dtype=np.float64)

    if t < 0 or t >= len(suffix_i):
        return out

    den = float(suffix_i[t])
    if den <= 0.0:
        return out

    if x_i == 0:
        out[:] = K_pmf_i
        return out

    for mu in range(11):
        rem = t - mu
        if 0 <= rem < len(suffix_ip1):
            out[mu] = K_pmf_i[mu] * float(suffix_ip1[rem]) / den

    s_out = out.sum()
    if s_out > 0.0:
        out /= s_out

    return out

def build_joint_coupling(kappa_og: np.ndarray, kappa_tilde: np.ndarray, p_mu: np.ndarray) -> np.ndarray:
    """
    Joint coupling matrix J where:
        J[mu1, mu2] = P(K_i = mu1, Ktilde_i = mu2 | ...)
    using the formulas you provided:

    Diagonal:
        P(K_i = Ktilde_i = mu) = min(kappa_og[mu], kappa_tilde[mu])

    Off-diagonal:
        P(K_i = mu1, Ktilde_i = mu2)
        = ((p_mu[mu2] * kappa_og[mu1] - p_mu[mu1] * kappa_tilde[mu2])_+) / (1 - p_mu[mu1])
        for mu1 != mu2
    """
    J = np.zeros((11, 11), dtype=np.float64)

    diag = np.minimum(kappa_og, kappa_tilde)
    np.fill_diagonal(J, diag)

    for mu1 in range(11):
        denom = 1.0 - p_mu[mu1]
        if denom <= 0.0:
            continue

        row_offdiag = np.maximum(p_mu * kappa_og[mu1] - p_mu[mu1] * kappa_tilde, 0.0) / denom
        row_offdiag[mu1] = 0.0
        J[mu1, :] += row_offdiag

    J = np.clip(J, 0.0, None)
    return J

def sample_categorical(prob: np.ndarray, rng: np.random.Generator) -> int:
    prob = np.asarray(prob, dtype=np.float64)
    s = prob.sum()
    if s <= 0.0:
        raise ValueError("Categorical probabilities sum to zero.")
    prob = prob / s
    c = np.cumsum(prob)
    c[-1] = 1.0
    u = float(rng.uniform(0.0, 1.0))
    return int(np.searchsorted(c, u, side="right"))

def coupling(K_pmf: np.ndarray, x: np.ndarray, r: int, r_tilde: int, rng: np.random.Generator, K_obs: np.ndarray) -> np.ndarray:
    """
    Construct the coupled next state K_tilde given the observed K_obs.
    """
    x = np.asarray(x, dtype=np.uint8)
    K_obs = np.asarray(K_obs, dtype=np.int16)

    n = x.size
    suffix = build_suffix_pmfs(K_pmf, x)

    s = 0
    s_tilde = 0
    K_tilde = np.empty(n, dtype=np.int16)

    for i in range(n):
        kappa_og = kappa(s, r, K_pmf[i], suffix[i], suffix[i + 1], int(x[i]))
        kappa_ti = kappa(s_tilde, r_tilde, K_pmf[i], suffix[i], suffix[i + 1], int(x[i]))

        J = build_joint_coupling(kappa_og, kappa_ti, K_pmf[i])

        observed_state = int(K_obs[i])
        row = J[observed_state, :]

        # In exact arithmetic, row.sum() should equal kappa_og[observed_state].
        # We renormalize to protect against small numerical drift.
        if row.sum() <= 0.0:
            # Deterministic fallback if the row is numerically empty.
            probs = np.zeros(11, dtype=np.float64)
            probs[observed_state] = 1.0
        else:
            probs = row / row.sum()

        K_tilde[i] = sample_categorical(probs, rng)

        s += int(K_obs[i]) * int(x[i])
        s_tilde += int(K_tilde[i]) * int(x[i])

    return K_tilde

@dataclass
class SDStepResult:
    U_j: float
    K_next: np.ndarray

def sd_one_step(K: np.ndarray, K_pmf: np.ndarray, x_col: np.ndarray, t_filter: float, rng: np.random.Generator) -> SDStepResult:
    K = np.asarray(K, dtype=np.int16)
    x = np.asarray(x_col, dtype=np.uint8)

    if K.shape[0] != x.shape[0]:
        raise ValueError("K and x_col must have the same length.")

    Sn = int(np.dot(K, x))

    suffix = build_suffix_pmfs(K_pmf, x)
    total_pmf = suffix[0]

    W, U, side, cdf = randomized_residual_and_pvalue(Sn, total_pmf, rng)

    U_prime = float(rng.uniform(0.0, 1.0))
    if U_prime <= 0.0:
        U_prime = float(EPS)

    U_j, Ue = simple_quantile_filter(U, U_prime, t_filter)
    Wf = (Ue / 2.0) if side == 1 else (1.0 - Ue / 2.0)

    if W == Wf:
        return SDStepResult(U_j, K)

    r = ppf_from_cdf(W, cdf)
    r_tilde = ppf_from_cdf(Wf, cdf)

    K_next = coupling(K_pmf, x, r, r_tilde, rng, K)
    return SDStepResult(U_j, K_next)

def run_sd(
    feather_path: str,
    y_prefix: str,
    p_hat_prefix: str,
    x_prefix: str,
    p_cols: int,
    t_filter: float,
    seed: int,
    progress_every: int,
    response_dim: int = 10,
) -> np.ndarray:
    if response_dim != 10:
        raise ValueError("This implementation assumes exactly 10 response coordinates.")

    df = pd.read_feather(feather_path)

    y_cols = [f"{y_prefix}{k}" for k in range(1, response_dim + 1)]
    p_hat_cols = [f"{p_hat_prefix}{k}" for k in range(1, response_dim + 1)]

    missing_y = [c for c in y_cols if c not in df.columns]
    missing_p = [c for c in p_hat_cols if c not in df.columns]
    if missing_y:
        raise KeyError(f"Missing response columns: {missing_y}")
    if missing_p:
        raise KeyError(f"Missing p_hat columns: {missing_p}")

    y_mat = df[y_cols].to_numpy(dtype=np.int8, copy=True)
    p_hat_logits = df[p_hat_cols].to_numpy(dtype=np.float64, copy=False)

    # Convert the observed 10-dim monotone binary response into K in {0,...,10}.
    K = y_matrix_to_K(y_mat)

    # Convert 10 logits into the 11-state PMF of K for each sample.
    K_pmf = logits_to_K_pmf(p_hat_logits)

    rng = np.random.default_rng(seed)
    U_out = np.empty(p_cols, dtype=np.float64)

    for j in range(p_cols):
        xname = f"{x_prefix}{j}"
        if xname not in df.columns:
            raise KeyError(f"Missing predictor column: {xname}")

        x_col = df[xname].to_numpy(dtype=np.uint8, copy=False)

        res = sd_one_step(K, K_pmf, x_col, t_filter, rng)
        U_out[j] = res.U_j
        K = res.K_next

        if progress_every and (j + 1) % progress_every == 0:
            print(f"Processed {j + 1}/{p_cols}.")

    return U_out

if __name__ == "__main__":
    U_vals = run_sd(
        feather_path="binary_test_data.feather",
        y_prefix="y",
        p_hat_prefix="p_hat",
        x_prefix="x",
        p_cols=60000,
        t_filter=0.001,
        seed=123,
        progress_every=10,
        response_dim=10,
    )
    print(U_vals)