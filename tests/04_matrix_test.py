import argparse
import ast
import datetime
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_triplets(filepath):
    triplets = []
    with open(filepath, 'r') as f:
        for line in f:
            a, b, c = line.split()
            triplets.append((int(a), int(b), int(c)))
    return triplets


def calculate_enhanced_stats(values):
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    outlier_low = q1 - 1.5 * iqr
    outlier_high = q3 + 1.5 * iqr
    outliers = [x for x in values if x < outlier_low or x > outlier_high]
    try:
        mode_result = stats.mode(values)
        mode = mode_result[0] if isinstance(mode_result[0], (int, float, np.number)) else mode_result[0][0]
    except (IndexError, AttributeError):
        mode = np.nan
    return {
        'mean': np.mean(values),
        'median': np.median(values),
        'mode': mode,
        'std': np.std(values),
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'outliers_count': len(outliers),
        'outliers_percent': len(outliers) / len(values) * 100 if len(values) else 0,
        'skewness': stats.skew(values),
        'kurtosis': stats.kurtosis(values)
    }


def plot_distribution(values, title, output_path):
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins='auto', density=True, alpha=0.7)
    plt.axvline(np.mean(values), color='red', linestyle='dashed', linewidth=1, label='Mean')
    plt.axvline(np.median(values), color='green', linestyle='dashed', linewidth=1, label='Median')
    plt.title(title)
    plt.xlabel('Margin')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()


def plot_boxplot(values, title, output_path):
    plt.figure(figsize=(8, 5))
    plt.boxplot(values)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()


def plot_qq(values, title, output_path):
    stats.probplot(values, dist="norm", plot=plt)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()


def evaluate_model(triplets, distance_matrix, target_margin=0.1):
    violations = 0
    triplet_margins = []
    margins = []
    abs_diffs = []

    for triplet in triplets:
        anchor, positive, negative = triplet
        pos_dist = distance_matrix[anchor, positive]
        neg_dist = distance_matrix[anchor, negative]

        margin = neg_dist - pos_dist
        abs_diff = abs(neg_dist - pos_dist)

        triplet_margins.append((margin, triplet, pos_dist, neg_dist, abs_diff))
        margins.append(margin)
        abs_diffs.append(abs_diff)

        if margin < target_margin:
            violations += 1

    triplet_margins.sort(key=lambda x: x[4])
    worst_cases = triplet_margins[:10]
    best_cases = triplet_margins[-10:]
    total = len(triplets)

    return {
        'total_triplets': total,
        'violations': violations,
        'violation_rate': violations / total if total else 0,
        'margins': margins,
        'abs_diffs': abs_diffs,
        'best_cases': best_cases,
        'worst_cases': worst_cases
    }


def print_enhanced_stats(stats, title="Enhanced Statistics"):
    print("\n" + title)
    print("=" * 40)
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Median: {stats['median']:.4f}")
    print(f"Mode: {stats['mode']:.4f}")
    print(f"Standard Deviation: {stats['std']:.4f}")
    print(f"Q1 (25th percentile): {stats['q1']:.4f}")
    print(f"Q3 (75th percentile): {stats['q3']:.4f}")
    print(f"IQR: {stats['iqr']:.4f}")
    print(f"Outliers: {stats['outliers_count']} ({stats['outliers_percent']:.2f}%)")
    print(f"Skewness: {stats['skewness']:.4f}")
    print(f"Kurtosis: {stats['kurtosis']:.4f}")


def print_cases(cases, title):
    print("\n" + title + ":")
    for margin, triplet, pos_sim, neg_sim, abs_diff in cases:
        print(
            f"Abs diff: {abs_diff:.4f}, Margin: {margin:.4f}, Triplet: {triplet}, Pos_sim: {pos_sim:.4f}, Neg_sim: {neg_sim:.4f}")


def aggregate_stats(all_stats):
    all_margins = []
    all_abs_diffs = []
    total_violations = 0
    total_triplets = 0
    for s in all_stats:
        all_margins.extend(s['margins'])
        all_abs_diffs.extend(s['abs_diffs'])
        total_violations += s['violations']
        total_triplets += s['total_triplets']
    margin_stats = calculate_enhanced_stats(all_margins)
    abs_diff_stats = calculate_enhanced_stats(all_abs_diffs)
    plot_distribution(all_margins, 'Distribution of Margins', 'margins_distribution.png')
    plot_distribution(all_abs_diffs, 'Distribution of Absolute Differences', 'abs_diffs_distribution.png')
    plot_boxplot(all_margins, 'Boxplot of Margins', 'margins_boxplot.png')
    plot_boxplot(all_abs_diffs, 'Boxplot of Absolute Differences', 'abs_diffs_boxplot.png')
    plot_qq(all_margins, 'QQ Plot of Margins', 'margins_qq.png')
    plot_qq(all_abs_diffs, 'QQ Plot of Absolute Differences', 'abs_diffs_qq.png')
    return {
        'total_triplets': total_triplets,
        'violations': total_violations,
        'violation_rate': total_violations / total_triplets if total_triplets else 0,
        'margin_stats': margin_stats,
        'abs_diff_stats': abs_diff_stats,
        'worst_cases': min((s['worst_cases'] for s in all_stats), key=lambda x: x[0][4]) if all_stats else [],
        'best_cases': max((s['best_cases'] for s in all_stats), key=lambda x: x[-1][4]) if all_stats else []
    }


def print_stats(stats, is_final=False):
    if not is_final:
        print(f"\nTotal triplets: {stats['total_triplets']}")
        print(f"Violations: {stats['violations']}")
        print(f"Violation rate: {stats['violation_rate']:.2%}")
        margin_stats = calculate_enhanced_stats(stats['margins'])
        abs_diff_stats = calculate_enhanced_stats(stats['abs_diffs'])
        print_enhanced_stats(margin_stats, "Margin Statistics")
        print_enhanced_stats(abs_diff_stats, "Absolute Difference Statistics")
        print_cases(stats['worst_cases'], "Worst cases (smallest absolute difference)")
        print_cases(stats['best_cases'], "Best cases (largest absolute difference)")
        return
    print("\nFINAL STATISTICS ACROSS ALL TESTS")
    print("=" * 40)
    print(f"Total triplets: {stats['total_triplets']}")
    print(f"Total violations: {stats['violations']}")
    print(f"Overall violation rate: {stats['violation_rate']:.2%}")
    print_enhanced_stats(stats['margin_stats'], "Margin Statistics")
    print_enhanced_stats(stats['abs_diff_stats'], "Absolute Difference Statistics")
    print_cases(stats['worst_cases'], "Global worst cases")
    print_cases(stats['best_cases'], "Global best cases")
    print("\nDistribution and box/QQ plots saved in the current directory")


def save_results_to_csv(all_stats):
    rows = []
    for i, s in enumerate(all_stats):
        rows.append({
            'file_index': i,
            'total_triplets': s['total_triplets'],
            'violations': s['violations'],
            'violation_rate': s['violation_rate']
        })
    df = pd.DataFrame(rows)
    df.to_csv('evaluation_summary.csv', index=False)


def save_anomalies(all_stats, output_file):
    with open(output_file, 'w') as f:
        for i, s in enumerate(all_stats):
            for margin, triplet, pos_sim, neg_sim, abs_diff in s['worst_cases']:
                f.write(
                    f"file_index:{i}, anchor:{triplet[0]}, positive:{triplet[1]}, negative:{triplet[2]}, abs_diff:{abs_diff:.4f}, margin:{margin:.4f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance-matrix', required=True)
    parser.add_argument('--triplets-dir', required=True)
    parser.add_argument('--target-margin', type=float, default=0.2)
    parser.add_argument('--save-anomalies', action='store_true')
    args = parser.parse_args()
    distance_matrix = np.load(args.distance_matrix)
    all_stats = []
    for file in glob(f'{args.triplets_dir}/*.txt'):
        print(f"\nProcessing {file}:")
        triplets = load_triplets(file)
        s = evaluate_model(triplets, distance_matrix, args.target_margin)
        all_stats.append(s)
        print_stats(s)
    final_stats = aggregate_stats(all_stats)
    print_stats(final_stats, is_final=True)
    save_results_to_csv(all_stats)
    if args.save_anomalies:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        anomalies_file = f"anomalies_{date_str}.txt"
        save_anomalies(all_stats, anomalies_file)


if __name__ == '__main__':
    main()
