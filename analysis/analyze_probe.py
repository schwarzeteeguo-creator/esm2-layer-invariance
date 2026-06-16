"""Analyze probing results directly."""
import paramiko, json, numpy as np

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westc.seetacloud.com', port=27439, username='root', password='ONn5cUuobkX2', timeout=15)

# Download results file via SFTP
sftp = ssh.open_sftp()
with sftp.open('/root/autodl-tmp/layer_probing_results/layer_probing_results.json', 'r') as f:
    results = json.load(f)
sftp.close()

print(f"Datasets: {len(results)}")
print(f"Dataset names: {list(results.keys())}\n")

# Per-layer aggregation
layer_rhos = {l: [] for l in range(33)}
for name, data in results.items():
    scores = data["layer_scores"]
    for l in range(33):
        s = scores[str(l)]["spearman"]
        if not np.isnan(s):
            layer_rhos[l].append(s)

print(f"{'Layer':>6} {'Mean Rho':>10} {'Std':>8} {'Min':>8} {'Max':>8} {'N':>5}")
print("-" * 55)
best_l, best_r = 32, -999
for l in range(33):
    if layer_rhos[l]:
        mean_r = np.mean(layer_rhos[l])
        std_r = np.std(layer_rhos[l])
        min_r = np.min(layer_rhos[l])
        max_r = np.max(layer_rhos[l])
        n = len(layer_rhos[l])
        marker = ""
        if l == 32:
            marker = " <-- final"
        if mean_r > best_r:
            best_r = mean_r
            best_l = l
        if l == best_l:
            marker += " <-- BEST"
        print(f"  {l:3d}   {mean_r:>10.4f} {std_r:>8.4f} {min_r:>8.4f} {max_r:>8.4f} {n:>5}{marker}")

final_r = np.mean(layer_rhos[32]) if layer_rhos[32] else 0
print("-" * 55)
print(f"\nBest layer:     {best_l} (rho = {best_r:.4f})")
print(f"Final layer:    32 (rho = {final_r:.4f})")
print(f"Delta:          {best_r - final_r:+.4f}")
if abs(final_r) > 1e-6:
    print(f"Relative gain:  {(best_r - final_r) / abs(final_r) * 100:+.1f}%")

# Per-dataset breakdown
print("\n\n=== Per-Dataset Layer Range ===")
print(f"{'Dataset':<50} {'Min Layer':>8} {'Min Rho':>8} {'Max Layer':>8} {'Max Rho':>8} {'Spread':>8}")
for name, data in sorted(results.items()):
    scores = data["layer_scores"]
    layer_vals = []
    for l in range(33):
        s = scores[str(l)]["spearman"]
        if not np.isnan(s):
            layer_vals.append((l, s))
    if layer_vals:
        min_pair = min(layer_vals, key=lambda x: x[1])
        max_pair = max(layer_vals, key=lambda x: x[1])
        spread = max_pair[1] - min_pair[1]
        print(f"  {name[:48]:<50} {min_pair[0]:>8} {min_pair[1]:>8.4f} {max_pair[0]:>8} {max_pair[1]:>8.4f} {spread:>8.4f}")

# Check process status and log
s, o, e = ssh.exec_command('tail -3 /root/autodl-tmp/probe_full.log 2>/dev/null | cat -v | head -5')
print("\n\n=== Log tail ===")
print(o.read().decode('utf-8', errors='replace')[:500])

s, o, e = ssh.exec_command('ps aux | grep "python.*layer_probing" | grep -v grep | wc -l')
print(f"\nProcesses running: {o.read().decode().strip()}")

ssh.close()
