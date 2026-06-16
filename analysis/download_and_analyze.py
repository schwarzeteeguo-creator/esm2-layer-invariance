"""Download latest results and run stats analysis."""
import paramiko, json, os, subprocess, sys

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westc.seetacloud.com', port=27439, username='root', password='ONn5cUuobkX2', timeout=30)

local_dir = r'C:\Users\19700\Desktop\蛋白质课题\results'
os.makedirs(local_dir, exist_ok=True)

PY = '/root/autodl-tmp/miniconda3/envs/saprot/bin/python'

# 1. Download benchmark results
print("1. Downloading benchmark results...")
sftp = ssh.open_sftp()
try:
    sftp.get('/root/autodl-tmp/benchmark_results_v2_saprot/multiscale_results_v2.json',
             os.path.join(local_dir, 'multiscale_results_v2.json'))
    print("   [OK] multiscale_results_v2.json")
except Exception as e:
    print(f"   [SKIP] {e}")
sftp.close()

# 2. Download benchmark summary if exists
print("\n2. Checking for summary...")
s, o, e = ssh.exec_command(f'ls /root/autodl-tmp/benchmark_results_v2_saprot/')
print(f"   Files: {o.read().decode('utf-8', errors='replace').strip()}")

# 3. Run quick analysis on the pooling data
print("\n3. Analyzing pooling results...")
pooling_path = os.path.join(local_dir, 'multiscale_results_v2.json')
if os.path.exists(pooling_path):
    with open(pooling_path, encoding='utf-8') as f:
        data = json.load(f)
    print(f"   {len(data)} datasets loaded")

    # Compute per-method means
    import numpy as np
    methods = ['last_layer', 'mean_20_33', 'concat_6_14_20_26_33', 'attention']
    readouts = ['ridge_rho', 'rf_rho']

    print("\n   === Pooling Method Comparison ===")
    print(f"   {'Method':<25} {'Ridge':>10} {'RF':>10}")
    print(f"   {'-'*45}")

    summary = {}
    for method in methods:
        ridge_vals = []
        rf_vals = []
        for dset_name, d in data.items():
            mdata = d.get('methods', {}).get(method, {})
            r = mdata.get('ridge_rho')
            f = mdata.get('rf_rho')
            if r is not None: ridge_vals.append(r)
            if f is not None: rf_vals.append(f)

        ridge_mean = np.mean(ridge_vals) if ridge_vals else 0
        ridge_std = np.std(ridge_vals) if ridge_vals else 0
        rf_mean = np.mean(rf_vals) if rf_vals else 0
        rf_std = np.std(rf_vals) if rf_vals else 0

        short_name = method.replace('concat_6_14_20_26_33', 'concat_5L')
        print(f"   {short_name:<25} {ridge_mean:>9.4f}  {rf_mean:>9.4f}")

        summary[method] = {'ridge_mean': ridge_mean, 'ridge_std': ridge_std,
                           'rf_mean': rf_mean, 'rf_std': rf_std, 'n': len(ridge_vals)}

    # Compute invariance
    ridge_means = [summary[m]['ridge_mean'] for m in methods]
    rf_means = [summary[m]['rf_mean'] for m in methods]
    print(f"\n   Ridge invariance: {max(ridge_means)-min(ridge_means):.2e}")
    print(f"   RF invariance: {max(rf_means)-min(rf_means):.2e}")

    # Save summary
    with open(os.path.join(local_dir, 'pooling_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\n   Saved: pooling_analysis.json")

# 4. Check benchmark progress
print("\n4. Benchmark progress...")
pid = ''
s, o, e = ssh.exec_command('pgrep -f benchmark_v2 | head -1')
pid = o.read().decode('utf-8', errors='replace').strip()
if pid:
    elapsed = ''
    s, o, e = ssh.exec_command(f'ps -o etime -p {pid} --no-headers')
    elapsed = o.read().decode('utf-8', errors='replace').strip()
    print(f"   PID {pid} | elapsed: {elapsed}")

s, o, e = ssh.exec_command(f'{PY} -c "import json; d=json.load(open(\\"/root/autodl-tmp/benchmark_results_v2_saprot/multiscale_results_v2.json\\")); print(len(d))"')
n = o.read().decode('utf-8', errors='replace').strip()
print(f"   Datasets processed: {n}")

ssh.close()
print("\n=== Done ===")
