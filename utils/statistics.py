# /utils/statistics.py
"""
Module for statistical analysis of simulation results.
"""
import numpy as np
from scipy import stats
from itertools import combinations

def _effect_magnitude_r(r):
    """Classifies rank-biserial correlation effect magnitude into common bins."""
    ar = abs(r)
    if ar >= 0.5: return 'large'
    if ar >= 0.3: return 'medium'
    if ar >= 0.1: return 'small'
    return 'trivial'

def perform_u_test_mannwhitney(all_results, alpha, verbose=True):
    """
    Runs the Mann-Whitney U test for all algorithm pairs.
    """
    algo_keys = list(all_results.keys())
    comparison_results = []

    for key_a, key_b in combinations(algo_keys, 2):
        data_a = np.asarray([m for m in all_results[key_a]['makespan'] if m != float('inf')])
        data_b = np.asarray([m for m in all_results[key_b]['makespan'] if m != float('inf')])

        result = {
            'algo_a': key_a, 'algo_b': key_b,
            'n_a': len(data_a), 'n_b': len(data_b)
        }

        if len(data_a) < 2 or len(data_b) < 2:
            if verbose:
                print(f"\nComparison {key_a} vs {key_b}: Not enough valid data.")
            comparison_results.append(result)
            continue
            
        u_stat, p_value = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
        rank_biserial_corr = 1 - (2 * u_stat) / (len(data_a) * len(data_b))

        result.update({
            'mean_a': np.mean(data_a), 'std_a': np.std(data_a, ddof=1),
            'mean_b': np.mean(data_b), 'std_b': np.std(data_b, ddof=1),
            'u_stat': u_stat, 'p_value': p_value,
            'rank_biserial_r': rank_biserial_corr,
            'effect_magnitude': _effect_magnitude_r(rank_biserial_corr),
            'is_significant': p_value < alpha,
            'better_algo': key_a if np.mean(data_a) < np.mean(data_b) else key_b
        })
        comparison_results.append(result)
        
        if verbose:
            print(f"\n--- Comparison: {key_a} vs {key_b} ---")
            print(f"  - {key_a}: Mean={result['mean_a']:.2f}, N={result['n_a']}")
            print(f"  - {key_b}: Mean={result['mean_b']:.2f}, N={result['n_b']}")
            
            p_display = '≥0.05' if p_value >= 0.05 else f'{p_value:.4f}'
            print(f"  - Mann-Whitney U Test: p-value = {p_display}")
            
            if result['is_significant']:
                print(f"  - Conclusion: Statistically significant difference. Better mean: {result['better_algo']}.")
            else:
                print("  - Conclusion: No statistically significant difference.")

    return comparison_results