"""Final analysis of all probing results for Bioinformatics paper."""
import paramiko, json, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westc.seetacloud.com', port=27439, username='root', password='ONn5cUuobkX2', timeout=15)

def r(cmd):
    s, o, e = ssh.exec_command(cmd)
    return o.read().decode('utf-8', errors='replace')

# Check completion
print('=== Process Status ===')
proc = r('ps aux | grep layer_probing_v2 | grep python | grep -v grep | wc -l')
print(f'Processes: {proc.strip()}')

count = r('grep -c n_mutations /root/autodl-tmp/layer_probing_results_v2/layer_probing_v2_results.json 2>/dev/null')
print(f'Datasets in JSON: {count.strip()}')

# Run comprehensive analysis on server
script = '''
import json, numpy as np
from scipy.stats import spearmanr, ttest_rel, wilcoxon

# Load probing results
d = json.load(open("/root/autodl-tmp/layer_probing_results_v2/layer_probing_v2_results.json"))
keys = list(d.keys())
N = len(keys)
print(f"Total datasets: {N}")

# Per-layer aggregation
num_layers = 33
layer_rhos = {l: [] for l in range(num_layers)}
layer_means_by_dataset = {l: [] for l in range(num_layers)}  # per-dataset mean for statistics

for name in keys:
    scores = d[name]["layer_scores"]
    dataset_vals = []
    for l in range(num_layers):
        s = scores[str(l)]["spearman"]
        if not np.isnan(s):
            layer_rhos[l].append(s)
            dataset_vals.append(s)
    # Per-dataset per-layer values for paired tests
    for l in range(num_layers):
        s = scores[str(l)]["spearman"]
        if not np.isnan(s):
            layer_means_by_dataset[l].append(s)

# Compute statistics
print(f"\\n{'Layer':>6} {'Mean Rho':>10} {'Std':>8} {'Median':>8} {'Min':>8} {'Max':>8} {'N':>5}")
print("-" * 65)
best_l, best_r = 32, -999
final_r = 0
for l in range(num_layers):
    if not layer_rhos[l]:
        continue
    vals = np.array(layer_rhos[l])
    mean_r = np.mean(vals)
    std_r = np.std(vals)
    median_r = np.median(vals)
    min_r = np.min(vals)
    max_r = np.max(vals)
    n = len(vals)
    marker = ""
    if l == 32:
        marker = " <-- final"
        final_r = mean_r
    if mean_r > best_r:
        best_r = mean_r
        best_l = l
    if l == best_l:
        marker += " <-- BEST"
    print(f"  {l:3d}   {mean_r:>10.4f} {std_r:>8.4f} {median_r:>8.4f} {min_r:>8.4f} {max_r:>8.4f} {n:>5}{marker}")

print("-" * 65)

# Paired t-test: best layer vs final layer
if best_l != 32:
    best_vals = np.array(layer_rhos[best_l])
    final_vals = np.array(layer_rhos[32])
    # Only use datasets where both are valid
    common_keys = []
    best_common = []
    final_common = []
    for name in keys:
        bs = d[name]["layer_scores"].get(str(best_l), {}).get("spearman", np.nan)
        fs = d[name]["layer_scores"].get("32", {}).get("spearman", np.nan)
        if not np.isnan(bs) and not np.isnan(fs):
            best_common.append(bs)
            final_common.append(fs)
    best_common = np.array(best_common)
    final_common = np.array(final_common)

    t_stat, p_val = ttest_rel(best_common, final_common)
    w_stat, w_p = wilcoxon(best_common, final_common)

    delta = best_r - final_r
    print(f"\\nBest layer:     {best_l} (rho = {best_r:.4f} +/- {np.std(best_vals):.4f})")
    print(f"Final layer:    32 (rho = {final_r:.4f} +/- {np.std(final_vals):.4f})")
    print(f"Delta:          {delta:+.4f}")
    print(f"Relative gain:  {delta/abs(final_r)*100:+.1f}%")
    print(f"Paired t-test:  t={t_stat:.4f}, p={p_val:.4f}")
    print(f"Wilcoxon:       W={w_stat:.1f}, p={w_p:.4f}")

    if p_val < 0.05:
        print("SIGNIFICANT difference (p < 0.05)")
    else:
        print("NOT significant (p >= 0.05)")
else:
    print(f"\\nBest layer IS the final layer (32): rho = {best_r:.4f}")

# Cross-dataset variance analysis
print(f"\\n=== Cross-Dataset Statistics ===")
all_dataset_means = []
for name in keys:
    vals = [d[name]["layer_scores"][str(l)]["spearman"] for l in range(num_layers)
            if not np.isnan(d[name]["layer_scores"][str(l)]["spearman"])]
    if vals:
        all_dataset_means.append(np.mean(vals))
all_dataset_means = np.array(all_dataset_means)
print(f"Dataset-level mean rho: {np.mean(all_dataset_means):.4f} +/- {np.std(all_dataset_means):.4f}")
print(f"Range: {np.min(all_dataset_means):.4f} - {np.max(all_dataset_means):.4f}")
print(f"Low performers (rho<0.2): {np.sum(all_dataset_means < 0.2)} datasets")
print(f"High performers (rho>0.5): {np.sum(all_dataset_means > 0.5)} datasets")

# Within-dataset layer spread
spreads = []
for name in keys:
    vals = [d[name]["layer_scores"][str(l)]["spearman"] for l in range(num_layers)
            if not np.isnan(d[name]["layer_scores"][str(l)]["spearman"])]
    if vals:
        spreads.append(np.max(vals) - np.min(vals))
spreads = np.array(spreads)
print(f"\\nWithin-dataset layer spread:")
print(f"  Mean:  {np.mean(spreads):.4f}")
print(f"  Median: {np.median(spreads):.4f}")
print(f"  Max:   {np.max(spreads):.4f}")
print(f"  % datasets with spread > 0.02: {np.sum(spreads > 0.02)/len(spreads)*100:.1f}%")

# Layer invariance score: std of per-layer means
layer_means = [np.mean(layer_rhos[l]) for l in range(num_layers) if layer_rhos[l]]
invariance_score = np.std(layer_means)
print(f"\\nLayer invariance score (std of per-layer means): {invariance_score:.6f}")
print(f"(Smaller = more invariant. 0 = perfectly invariant)")

# Total mutations
total_muts = sum(d[name]["n_mutations"] for name in keys)
print(f"\\nTotal mutations across all datasets: {total_muts:,}")
'''

cmd = 'cd /root/autodl-tmp && eval "$(/root/autodl-tmp/miniconda3/bin/conda shell.bash hook)" && conda activate saprot && python -c \'{}\''.format(script)
print('\n=== PROBING ANALYSIS ===')
print(r(cmd))

# Load similarity data
print('\n=== LAYER SIMILARITY ===')
sftp = ssh.open_sftp()
with sftp.open('/root/autodl-tmp/layer_similarity_results/layer_similarity.json', 'r') as f:
    sim_data = json.load(f)
sftp.close()

print(f"Datasets analyzed: {sim_data['num_datasets']}")
print(f"Mean neighbor similarity: {sim_data['stats']['mean_neighbor_sim']:.4f}")
print(f"Layer 0 vs 1:  {sim_data['stats']['layer0_vs_1']:.4f}")
print(f"Layer 0 vs 16: {sim_data['stats']['layer0_vs_16']:.4f}")
print(f"Layer 0 vs 32: {sim_data['stats']['layer0_vs_32']:.4f}")
print(f"Layer 16 vs 32: {sim_data['stats']['layer16_vs_32']:.4f}")
print(f"Distance-similarity correlation: r={sim_data['stats']['distance_sim_correlation']:.4f}")

ssh.close()
