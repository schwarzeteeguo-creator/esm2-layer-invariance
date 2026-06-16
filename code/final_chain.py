"""Final chain: ESM-2 Ridge+RF → ESM-2 MLP+LGBM → SaProt MLP+LGBM → download all"""
import paramiko, json, os, time, numpy as np

LOCAL = r'C:\Users\19700\Desktop\蛋白质课题\results'
os.makedirs(LOCAL, exist_ok=True)

def connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('connect.westc.seetacloud.com', port=27439, username='root', password='ONn5cUuobkX2', timeout=30)
    return ssh

PY = '/root/autodl-tmp/miniconda3/envs/saprot/bin/python'
log_file = os.path.join(os.path.dirname(__file__), 'final_chain_log.txt')

def log(msg):
    t = time.strftime('%H:%M:%S')
    line = f"[{t}] {msg}"
    print(line)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def count_ridge(path):
    """Count datasets with valid Ridge results"""
    ssh = connect()
    sftp = ssh.open_sftp()
    try:
        with sftp.open(path, 'r') as f:
            d = json.loads(f.read().decode('utf-8'))
        sftp.close()
        n = sum(1 for v in d.values() if v['methods']['last_layer'].get('ridge_rho') is not None)
        ssh.close()
        return n
    except:
        sftp.close()
        ssh.close()
        return 0

def wait_for_results(path, target=20, label=''):
    while True:
        n = count_ridge(path)
        log(f"  {label}: {n}/{target}")
        if n >= target:
            break
        time.sleep(180)  # 3 min

def run_benchmark(model_key, model_dir, output_dir, readout_models, log_name):
    ssh = connect()
    cmd = (
        f'cd /root/autodl-tmp && nohup {PY} -m esm_embedding.benchmark_v2 '
        f'--data_dir /root/autodl-tmp/SaProt/LMDB/ProteinGym/substitutions '
        f'--model_dir {model_dir} --model_key {model_key} '
        f'--output_dir {output_dir} --device cuda --max_datasets 20 '
        f'--readout_models {readout_models} '
        f'>> /root/autodl-tmp/{log_name} 2>&1 &'
    )
    s, o, e = ssh.exec_command(cmd)
    time.sleep(10)
    ssh.close()

def download(remote_path, local_name):
    ssh = connect()
    sftp = ssh.open_sftp()
    local_path = os.path.join(LOCAL, local_name)
    sftp.get(remote_path, local_path)
    sftp.close()
    ssh.close()
    return local_path

def analyze(path, model_name):
    with open(path) as f:
        d = json.load(f)
    methods = ['last_layer', 'mean_20_33', 'concat_6_14_20_26_33', 'attention']
    readouts = ['ridge_rho', 'rf_rho', 'mlp_rho', 'lightgbm_rho']
    labels = ['Ridge', 'RF', 'MLP', 'LGBM']

    log(f"\n=== {model_name} ===")
    header = f"{'Method':<30}"
    for lb in labels:
        header += f" {lb:>8}"
    log(header)
    log('-' * 62)
    for method in methods:
        row = f"{method[:30]:<30}"
        for rk in readouts:
            vals = [v['methods'][method][rk] for v in d.values()
                    if v.get('methods', {}).get(method, {}).get(rk) is not None]
            row += f" {np.mean(vals):>8.4f}" if vals else f" {'N/A':>8}"
        log(row)

    for lb, rk in zip(labels, readouts):
        means = []
        for method in methods:
            vals = [v['methods'][method][rk] for v in d.values()
                    if v.get('methods', {}).get(method, {}).get(rk) is not None]
            if vals: means.append(np.mean(vals))
        if len(means) >= 2:
            log(f"  {lb} invariance: {max(means)-min(means):.2e}")

# ============================================================
log("=== FINAL CHAIN STARTED ===")

# Step 1: Wait for ESM-2 Ridge+RF (already running)
log("Step 1: Waiting for ESM-2 Ridge+RF (should be running)...")
esm2_file = '/root/autodl-tmp/benchmark_results_v2_esm2/multiscale_results_v2.json'
wait_for_results(esm2_file, 20, 'ESM-2 Ridge+RF')
log("Step 1 DONE: ESM-2 Ridge+RF 20/20")

# Step 2: Download & analyze ESM-2 Ridge+RF
log("Step 2: Downloading ESM-2 Ridge+RF...")
download(esm2_file, 'multiscale_results_v2_esm2.json')
analyze(os.path.join(LOCAL, 'multiscale_results_v2_esm2.json'), 'ESM-2 (Ridge+RF)')

# Step 3: Run ESM-2 MLP+LGBM (merge into existing)
log("Step 3: Running ESM-2 MLP+LGBM merge...")
run_benchmark('esm2_650m', '/root/autodl-tmp/esm2_model_built',
              '/root/autodl-tmp/benchmark_results_v2_esm2',
              'mlp lightgbm', 'esm2_mlp_final.log')

# Step 4: Wait for ESM-2 MLP+LGBM
log("Step 4: Waiting for ESM-2 MLP+LGBM...")
wait_for_results(esm2_file, 20, 'ESM-2 MLP+LGBM (checking mlp)')

# But we need to check MLP not Ridge! Ridge is already done.
# Wait for process to exit
while True:
    ssh = connect()
    s, o, e = ssh.exec_command('ps aux | grep "esm2_650m.*mlp lightgbm" | grep -v grep | wc -l')
    running = int(o.read().decode().strip())
    ssh.close()
    log(f"  ESM-2 MLP+LGBM running: {running}")
    if running == 0:
        break
    time.sleep(300)

log("Step 4 DONE: ESM-2 MLP+LGBM")

# Step 5: Download final ESM-2
log("Step 5: Downloading final ESM-2...")
download(esm2_file, 'multiscale_results_v2_esm2.json')
analyze(os.path.join(LOCAL, 'multiscale_results_v2_esm2.json'), 'ESM-2 FINAL')

# Step 6: Run SaProt MLP+LGBM
log("Step 6: Running SaProt MLP+LGBM...")
run_benchmark('saprot_650m', '/root/autodl-tmp/SaProt/weights/PLMs/SaProt_650M_AF2_hf',
              '/root/autodl-tmp/benchmark_results_v2_saprot',
              'mlp lightgbm', 'saprot_mlp_final.log')

# Step 7: Wait for SaProt MLP+LGBM
log("Step 7: Waiting for SaProt MLP+LGBM...")
saprot_file = '/root/autodl-tmp/benchmark_results_v2_saprot/multiscale_results_v2.json'
while True:
    ssh = connect()
    s, o, e = ssh.exec_command('ps aux | grep "saprot.*mlp lightgbm" | grep -v grep | wc -l')
    running = int(o.read().decode().strip())
    sftp = ssh.open_sftp()
    try:
        with sftp.open(saprot_file, 'r') as f:
            d = json.loads(f.read().decode('utf-8'))
        mlp = sum(1 for v in d.values() if v['methods']['last_layer'].get('mlp_rho') is not None)
    except:
        mlp = 0
    sftp.close()
    ssh.close()
    log(f"  SaProt MLP+LGBM: mlp={mlp}/20 running={running}")
    if running == 0:
        break
    time.sleep(300)

log("Step 7 DONE: SaProt MLP+LGBM")

# Step 8: Download final SaProt
log("Step 8: Downloading final SaProt...")
download(saprot_file, 'multiscale_results_v2.json')
analyze(os.path.join(LOCAL, 'multiscale_results_v2.json'), 'SaProt FINAL')

# Step 9: Regenerate figures
log("Step 9: Regenerating figures...")
os.chdir(os.path.join(os.path.dirname(__file__), 'figures'))
os.system('python generate_figures.py')

log("\n=== ALL DONE! Phase A complete. ===")
