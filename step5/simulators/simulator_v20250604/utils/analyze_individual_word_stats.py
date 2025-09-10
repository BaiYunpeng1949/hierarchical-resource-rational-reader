import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List, Optional

def _is_discrete_like(x: np.ndarray) -> bool:
    """Heuristic: treat as discrete if all finite values are integers and unique count is small-ish."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return False
    return np.allclose(x, np.rint(x)) and np.unique(x).size <= max(100, int(0.2 * x.size))

def _fit_distribution_continuous(x: np.ndarray, dist_name: str):
    dist = getattr(stats, dist_name)
    params = dist.fit(x)
    # params = (shape(s)?, loc, scale)
    # log-likelihood
    ll = np.sum(dist.logpdf(x, *params))
    k = len(params)
    aic = 2 * k - 2 * ll
    # KS test (uses fitted params)
    ks_stat, ks_p = stats.kstest(x, dist_name, args=params)
    return {
        "name": dist_name,
        "params": params,
        "loglik": float(ll),
        "aic": float(aic),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "type": "continuous",
    }

def _fit_distribution_discrete(x: np.ndarray, dist_name: str):
    dist = getattr(stats, dist_name)
    # For discrete distributions, scipy's fit often expects integers
    x_int = np.asarray(np.rint(x), dtype=int)
    params = dist.fit(x_int)
    # log-likelihood via pmf
    ll = np.sum(dist.logpmf(x_int, *params))
    k = len(params)
    aic = 2 * k - 2 * ll
    # One-sample KS is for continuous; use a simple CDF-based discrete variant (Dwass–Stephens)
    # We'll approximate with the classical KS against the CDF at midpoints.
    # (Good enough as a comparative heuristic; for rigorous tests, consider chi-square.)
    sorted_x = np.sort(x_int)
    cdf_vals = dist.cdf(sorted_x, *params)
    emp_cdf = np.arange(1, len(sorted_x)+1) / len(sorted_x)
    ks_stat = float(np.max(np.abs(emp_cdf - cdf_vals)))
    ks_p = np.nan  # not computed here
    return {
        "name": dist_name,
        "params": params,
        "loglik": float(ll),
        "aic": float(aic),
        "ks_stat": ks_stat,
        "ks_p": ks_p,
        "type": "discrete",
    }

def _plot_fit(x: np.ndarray, best: Dict[str, Any], bins: Optional[int] = None):
    is_discrete = best["type"] == "discrete"
    dist = getattr(stats, best["name"])
    params = best["params"]

    if is_discrete:
        # Bar plot of empirical frequencies
        vals, counts = np.unique(np.asarray(np.rint(x), dtype=int), return_counts=True)
        probs = counts / counts.sum()
        plt.figure()
        plt.bar(vals, probs, alpha=0.6, label="Empirical")
        # Overlay PMF
        xs = np.arange(vals.min(), vals.max()+1)
        pmf = dist.pmf(xs, *params)
        plt.plot(xs, pmf, linewidth=2, label=f"{best['name']} PMF")
        plt.title(f"Discrete data with {best['name']} fit")
        plt.legend()
        plt.xlabel("Value")
        plt.ylabel("Probability")
        plt.show()
    else:
        # Histogram + PDF
        plt.figure()
        plt.hist(x, bins=(bins or "auto"), density=True, alpha=0.6)
        xmin, xmax = plt.xlim()
        grid = np.linspace(xmin, xmax, 400)
        pdf = dist.pdf(grid, *params)
        plt.plot(grid, pdf, linewidth=2, label=f"{best['name']} PDF")
        plt.title(f"Histogram with best fit: {best['name']}")
        plt.legend()
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.show()

def sample_from_fit(best_fit: Dict[str, Any], n: int, random_state: Optional[int] = None) -> np.ndarray:
    """
    Draw n samples from a previously returned best_fit dictionary.
    """
    rng = np.random.default_rng(random_state)
    dist = getattr(stats, best_fit["name"])
    # scipy rvs ignores numpy Generator; we’ll set global seed only if provided
    if random_state is not None:
        np.random.seed(random_state)
    return dist.rvs(*best_fit["params"], size=n)

def analyze_distribution(
    values: List[float],
    *,
    try_continuous: Tuple[str, ...] = ("norm", "lognorm", "gamma", "expon", "weibull_min", "logistic"),
    try_discrete: Tuple[str, ...] = ("poisson", "nbinom"),
    force_discrete: Optional[bool] = None,
    plot: bool = True,
    bins: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze a list/array of values:
      - summary stats
      - best-fitting distribution by AIC (among candidates)
      - KS statistics (continuous exact; discrete approximate)
      - optional plot of histogram + best fit

    Returns a dict with:
      {
        'summary': pd.Series,
        'candidates': pd.DataFrame (sorted by AIC asc),
        'best_fit': dict(name, params, aic, ks_stat, ks_p, type),
        'is_discrete': bool
      }
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]  # drop NaN/inf

    # Summary stats
    summary = pd.Series({
        "count": x.size,
        "mean": np.mean(x) if x.size else np.nan,
        "median": np.median(x) if x.size else np.nan,
        "std": np.std(x, ddof=1) if x.size > 1 else np.nan,
        "var": np.var(x, ddof=1) if x.size > 1 else np.nan,
        "min": np.min(x) if x.size else np.nan,
        "q25": np.percentile(x, 25) if x.size else np.nan,
        "q75": np.percentile(x, 75) if x.size else np.nan,
        "max": np.max(x) if x.size else np.nan,
        "skew": stats.skew(x) if x.size >= 3 else np.nan,
        "kurtosis": stats.kurtosis(x) if x.size >= 4 else np.nan,
    })

    # Choose discrete/continuous path
    if force_discrete is None:
        is_discrete = _is_discrete_like(x)
    else:
        is_discrete = bool(force_discrete)

    results = []
    candidates = try_discrete if is_discrete else try_continuous

    for name in candidates:
        try:
            if is_discrete:
                fit = _fit_distribution_discrete(x, name)
            else:
                fit = _fit_distribution_continuous(x, name)
            results.append(fit)
        except Exception:
            # fitting can fail for some shapes/data; just skip
            continue

    if not results:
        print(f"The summary is: \n{summary}")
        # raise RuntimeError("No candidate distributions could be fitted to the data.")
        print(f"No candidate distributions could be fitted to the data.")

    if results:
        # Rank by AIC
        results_sorted = sorted(results, key=lambda d: d["aic"])
        best = results_sorted[0]

        # Optional plot
        if plot:
            try:
                _plot_fit(x, best, bins=bins)
            except Exception:
                # plotting failure shouldn't break analysis
                pass

        # Package outputs
        out = {
            "summary": summary,
            "candidates": pd.DataFrame(results_sorted),
            "best_fit": best,
            "is_discrete": is_discrete,
        }
        print(f"\n\nData distribution analysis is:\n{out}")
        return out
    else:
        return

# --- Example usage ---
if __name__ == "__main__":
    values = [10, 12, 13, 15, 20, 22, 22, 25, 30, 30, 35]
    report = analyze_distribution(values, plot=True)
    # print("Summary:\n", report["summary"])
    # print("\nCandidates (best at top by AIC):\n", report["candidates"])
    # print("\nBest fit:", report["best_fit"])
    # # Simulate 1000 samples from the best fit
    # sim = sample_from_fit(report["best_fit"], n=1000, random_state=42)
    # print("\nSimulated sample mean:", sim.mean())
