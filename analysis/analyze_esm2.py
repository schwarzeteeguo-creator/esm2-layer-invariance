import json, numpy as np

with open(r'C:\Users\19700\Desktop\蛋白质课题\results\multiscale_results_v2.json') as f:
    saprot = json.load(f)
with open(r'C:\Users\19700\Desktop\蛋白质课题\results\multiscale_results_v2_esm2.json') as f:
    esm2 = json.load(f)

common = sorted(set(saprot.keys()) & set(esm2.keys()))
print(f'Common datasets: {len(common)}')

print(f'\n{"Dataset":<38} {"ESM2-Last":>9} {"ESM2-Concat":>11} {"SaP-Last":>8} {"ESM2-RF":>8}')
print('-' * 78)
diffs = []
for name in common[:20]:
    e_last = esm2[name]['methods']['last_layer'].get('ridge_rho')
    e_concat = esm2[name]['methods']['concat_6_14_20_26_33'].get('ridge_rho')
    s_last = saprot[name]['methods']['last_layer'].get('ridge_rho')
    e_rf = esm2[name]['methods']['last_layer'].get('rf_rho')

    if e_last and e_concat:
        diff = e_concat - e_last
        diffs.append(diff)
        short = name[:36]
        print(f'{short:<38} {e_last:>9.4f} {e_concat:>11.4f} {s_last:>8.4f} {e_rf:>8.4f}')

print(f'\nESM-2 Ridge: Concat > Last in {sum(1 for d in diffs if d>0)}/{len(diffs)} datasets')
print(f'Mean diff (Concat - Last): {np.mean(diffs):.4f}')
print(f'Median diff: {np.median(diffs):.4f}')

# Check: is ESM-2 RF also concat > last?
rf_diffs = []
for name in common:
    e_rf_last = esm2[name]['methods']['last_layer'].get('rf_rho')
    e_rf_concat = esm2[name]['methods']['concat_6_14_20_26_33'].get('rf_rho')
    if e_rf_last and e_rf_concat:
        rf_diffs.append(e_rf_concat - e_rf_last)
print(f'\nESM-2 RF: Concat > Last in {sum(1 for d in rf_diffs if d>0)}/{len(rf_diffs)} datasets')
print(f'Mean RF diff: {np.mean(rf_diffs):.4f}')

# Check SaProt Ridge concat vs last
sp_diffs = []
for name in common:
    sp_last = saprot[name]['methods']['last_layer'].get('ridge_rho')
    sp_concat = saprot[name]['methods']['concat_6_14_20_26_33'].get('ridge_rho')
    if sp_last and sp_concat:
        sp_diffs.append(sp_concat - sp_last)
print(f'\nSaProt Ridge: Concat > Last in {sum(1 for d in sp_diffs if d>0)}/{len(sp_diffs)} datasets')
print(f'Mean SaProt diff: {np.mean(sp_diffs):.4f}')
