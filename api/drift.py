import numpy as np
try:
    from scipy import stats
except ImportError:
    stats = None

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """
    Calculate the Population Stability Index (PSI) between expected (reference)
    and actual (production) distributions.
    
    PSI Guidelines:
    - PSI < 0.1: No significant shift
    - 0.1 <= PSI < 0.25: Moderate shift
    - PSI >= 0.25: Significant shift
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    # Get bin edges based on expected distribution percentiles
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, percentiles)
    
    # Adjust boundaries to catch edge values
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    
    # Calculate counts and proportions
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    expected_pcts = expected_counts / len(expected)
    actual_pcts = actual_counts / len(actual)
    
    # Handle zero division / log issues
    expected_pcts = np.where(expected_pcts == 0, 1e-4, expected_pcts)
    actual_pcts = np.where(actual_pcts == 0, 1e-4, actual_pcts)
    
    # Calculate PSI
    psi_value = np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts))
    return float(psi_value)

def calculate_ks(expected: np.ndarray, actual: np.ndarray) -> dict:
    """
    Calculate the Kolmogorov-Smirnov (KS) 2-sample statistic and p-value.
    Used to detect if the distribution of incoming features has shifted.
    """
    if stats is None:
        return {"statistic": 0.0, "p_value": 1.0, "note": "scipy is not installed"}
        
    res = stats.ks_2samp(expected, actual)
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue)
    }
