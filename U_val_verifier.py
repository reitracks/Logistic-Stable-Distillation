import numpy as np
from math import log
import scipy.stats as stats
from binary_x_v2 import run_sd_fast

TARGET_FILE = "binary_20k_v2_19.feather"
FACTORS = 60000

def modified_renyi_outlier_test(U, tau):
    sorted_U = np.sort(U)
    filtered_U = sorted_U[sorted_U < tau]
    norm_U = filtered_U/tau
    norm_U = np.clip(norm_U, np.nextafter(0.0, 1.0), 1.0)

    if len(norm_U) == 0:
        return 1.0

    l1_norm = 0
    for i in range(len(norm_U)-1):
        l1_norm += abs((i+1)*log(norm_U[i+1]/norm_U[i]))
    l1_norm += abs(len(norm_U)*log(norm_U[-1]))

    binomial_cdf = stats.binom.cdf(len(norm_U), len(U), tau)
    
    gamma_in = l1_norm - log(np.clip(1-binomial_cdf, np.nextafter(0, 1), 1))
    gamma_cdf = stats.gamma.cdf(gamma_in, a = len(norm_U)+1)

    return 1 - gamma_cdf

def threshold(exponent, n = 60000, t_one_error_rate=1/60000):
    return stats.beta.ppf(1-2*t_one_error_rate, 2**exponent, n-2**exponent+1)

if __name__ == "__main__":
    p_vals=[0]*7
    u_star = run_sd_fast(TARGET_FILE, "y", "p_hat", "x", FACTORS, 1/FACTORS, 123, 10000)
    bonfferoni_p_val = min(u_star)*FACTORS
    print(f"Bonfferoni p value is {bonfferoni_p_val}")
    for i in range(2, 9):
        t_val = threshold(i, FACTORS)
        print(f"t_val is {t_val}")
        u_vals = run_sd_fast(TARGET_FILE, "y", "p_hat", "x", FACTORS, t_val, 123, 10000)
        #u_vals = np.random.random(size = 30000)
        #print(u_vals[:50])
        p_val = modified_renyi_outlier_test(u_vals, t_val)
        print(f"p_val is {p_val}")
        p_vals[i-2] = p_val
    print(f"List of p_values {p_vals}")
    p_vals.append(bonfferoni_p_val)
    print(f"Final p_values {min(min(p_vals)*8, 1)}")
    ks_test = stats.kstest(p_vals, stats.uniform.cdf)
    print(ks_test)

    