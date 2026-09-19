# Revision 主表（自动生成）

## 1. 位置泄漏控制（随机位置向量查找表，本地 GCV 协议，n=63）

| 特征 | random | modulo | contiguous | GroupKFold |
|---|---|---|---|---|
| 随机位置向量 + aux | +0.600 (n=63) | +0.137 (n=63) | +0.096 (n=57) | +0.135 (n=63) |
| 随机位置向量 alone | +0.557 (n=63) | -0.008 (n=63) | -0.056 (n=57) | -0.010 (n=63) |
| aux alone | +0.338 (n=63) | +0.230 (n=63) | +0.188 (n=57) | +0.228 (n=63) |

random vs modulo 配对 Wilcoxon（随机位置向量+aux）:

- n=63, 平均跌落 +0.463, Wilcoxon p=5.2e-12

## 2. Probing 主表（三臂 × 四划分；combined 特征集；格式：均值(最优层→L32)）

| 模型 | 臂 | random | modulo | contiguous | GroupKFold |
|---|---|---|---|---|---|
| ESM-2 | masked_wt | +0.606 (L32→+0.606) n=63 | +0.459 (L32→+0.459) n=63 | +0.370 (L32→+0.370) n=57 | +0.450 (L32→+0.450) n=63 |
| ESM-2 | unmasked_wt | +0.606 (L32→+0.606) n=63 | +0.470 (L32→+0.470) n=63 | +0.388 (L32→+0.388) n=57 | +0.461 (L32→+0.461) n=63 |
| ESM-2 | mutant | +0.681 (L32→+0.681) n=63 | +0.483 (L31→+0.479) n=63 | +0.365 (L31→+0.359) n=57 | +0.477 (L31→+0.475) n=63 |
| SaProt (struct-masked) | masked_wt | +0.603 (L31→+0.603) n=63 | +0.391 (L30→+0.386) n=63 | +0.282 (L31→+0.274) n=57 | +0.385 (L31→+0.382) n=63 |
| SaProt (struct-masked) | unmasked_wt | +0.604 (L32→+0.604) n=63 | +0.412 (L30→+0.406) n=63 | +0.324 (L31→+0.320) n=57 | +0.403 (L27→+0.400) n=63 |
| SaProt (struct-masked) | mutant | +0.652 (L23→+0.651) n=63 | +0.465 (L30→+0.460) n=63 | +0.323 (L32→+0.323) n=57 | +0.461 (L30→+0.458) n=63 |
| SaProt (real 3Di) | masked_wt | +0.606 (L32→+0.606) n=63 | +0.461 (L31→+0.460) n=63 | +0.394 (L31→+0.394) n=57 | +0.454 (L31→+0.453) n=63 |
| SaProt (real 3Di) | unmasked_wt | +0.606 (L27→+0.605) n=63 | +0.473 (L31→+0.466) n=63 | +0.415 (L31→+0.406) n=57 | +0.467 (L31→+0.459) n=63 |
| SaProt (real 3Di) | mutant | +0.684 (L26→+0.683) n=63 | +0.531 (L31→+0.525) n=63 | +0.451 (L31→+0.445) n=57 | +0.532 (L31→+0.524) n=63 |

## 3. TOST 等效检验（leakage-free 划分；固定层=均值曲线最优层，与 L32 配对；边际 Δρ=0.01）

| 模型 | 臂 | 划分 | 最优层 | n | mean Δ(best−final) | 90% CI | TOST p | 结论 |
|---|---|---|---|---|---|---|---|---|
| ESM-2 | mutant | modulo | L31 | 63 | +0.004 | [-0.003, +0.011] | 0.049 | 等效 |
| ESM-2 | mutant | contiguous | L31 | 57 | +0.006 | [-0.006, +0.019] | 0.250 | 不确定 |
| ESM-2 | mutant | position_groupkfold | L31 | 63 | +0.003 | [-0.006, +0.012] | 0.059 | 不确定 |
| SaProt (struct-masked) | masked_wt | modulo | L30 | 63 | +0.005 | [+0.000, +0.011] | 0.045 | 等效 |
| SaProt (struct-masked) | masked_wt | contiguous | L31 | 57 | +0.008 | [+0.003, +0.013] | 0.188 | 不确定 |
| SaProt (struct-masked) | masked_wt | position_groupkfold | L31 | 63 | +0.003 | [-0.000, +0.007] | 0.001 | 等效 |
| SaProt (struct-masked) | unmasked_wt | modulo | L30 | 63 | +0.006 | [+0.000, +0.013] | 0.136 | 不确定 |
| SaProt (struct-masked) | unmasked_wt | contiguous | L31 | 57 | +0.004 | [-0.002, +0.011] | 0.041 | 等效 |
| SaProt (struct-masked) | unmasked_wt | position_groupkfold | L27 | 63 | +0.003 | [-0.005, +0.011] | 0.052 | 不确定 |
| SaProt (struct-masked) | mutant | modulo | L30 | 63 | +0.005 | [-0.003, +0.013] | 0.098 | 不确定 |
| SaProt (struct-masked) | mutant | position_groupkfold | L30 | 63 | +0.004 | [-0.006, +0.014] | 0.110 | 不确定 |
| SaProt (real 3Di) | masked_wt | modulo | L31 | 63 | +0.002 | [-0.002, +0.005] | 0.000 | 等效 |
| SaProt (real 3Di) | masked_wt | contiguous | L31 | 57 | +0.001 | [-0.005, +0.007] | 0.001 | 等效 |
| SaProt (real 3Di) | masked_wt | position_groupkfold | L31 | 63 | +0.002 | [-0.002, +0.005] | 0.000 | 等效 |
| SaProt (real 3Di) | unmasked_wt | modulo | L31 | 63 | +0.007 | [+0.003, +0.011] | 0.081 | 不确定 |
| SaProt (real 3Di) | unmasked_wt | contiguous | L31 | 57 | +0.009 | [+0.003, +0.015] | 0.328 | 不确定 |
| SaProt (real 3Di) | unmasked_wt | position_groupkfold | L31 | 63 | +0.008 | [+0.004, +0.013] | 0.254 | 不确定 |
| SaProt (real 3Di) | mutant | modulo | L31 | 63 | +0.006 | [+0.000, +0.012] | 0.101 | 不确定 |
| SaProt (real 3Di) | mutant | contiguous | L31 | 57 | +0.006 | [-0.005, +0.016] | 0.218 | 不确定 |
| SaProt (real 3Di) | mutant | position_groupkfold | L31 | 63 | +0.008 | [+0.002, +0.014] | 0.240 | 不确定 |

## 4. SaProt 真3Di vs 结构掩码（同数据集同层配对，L32，combined）

| 臂 | 划分 | n | mean Δ(3di−mask) | Wilcoxon p |
|---|---|---|---|---|
| masked_wt | random | 63 | +0.003 | 2.2e-02 |
| masked_wt | modulo | 63 | +0.073 | 3.5e-07 |
| masked_wt | contiguous | 57 | +0.120 | 8.7e-08 |
| masked_wt | position_groupkfold | 63 | +0.071 | 8.4e-07 |
| unmasked_wt | random | 63 | +0.002 | 9.1e-01 |
| unmasked_wt | modulo | 63 | +0.060 | 4.6e-06 |
| unmasked_wt | contiguous | 57 | +0.086 | 2.1e-06 |
| unmasked_wt | position_groupkfold | 63 | +0.059 | 1.2e-05 |
| mutant | random | 63 | +0.032 | 2.5e-06 |
| mutant | modulo | 63 | +0.064 | 6.9e-05 |
| mutant | contiguous | 57 | +0.122 | 6.5e-07 |
| mutant | position_groupkfold | 63 | +0.067 | 4.6e-05 |

## 5. Logit lens（零样本，n=63，门禁全过）

- **ESM-2**: L0 +0.031 → 最优 L32 **+0.443** → L32 +0.443；曲线（每4层采样）: +0.03 +0.04 +0.02 +0.11 +0.11 +0.15 +0.24 +0.34 +0.44
- **SaProt real-3Di**: L0 -0.077 → 最优 L32 **+0.437** → L32 +0.437；曲线（每4层采样）: -0.08 -0.07 -0.05 -0.03 -0.00 +0.04 +0.09 +0.16 +0.44

## 6. CKA（Kornblith 线性 CKA vs cosine，10 数据集 × 50 位置）

- **ESM-2** (n=10): 相邻层 CKA 0.967 / cos 0.962；L0↔L32 CKA 0.554 / cos 0.558
- **SaProt** (n=10): 相邻层 CKA 0.979 / cos 0.979；L0↔L32 CKA 0.808 / cos -0.009

## 7. Pooling（n=20/模型设计；本节随进度自动刷新）

### ESM-2（已完成 20/20）
| 策略 | 读出 | mutant@modulo ρ |
|---|---|---|
| last_layer | ridge | +0.463 (n=20) |
| last_layer | mlp | +0.480 (n=20) |
| last_layer | lightgbm | +0.508 (n=20) |
| mean_20_33 | ridge | +0.473 (n=20) |
| mean_20_33 | mlp | +0.436 (n=20) |
| mean_20_33 | lightgbm | +0.457 (n=20) |
| concat_5L | ridge | +0.481 (n=20) |
| concat_5L | mlp | +0.435 (n=20) |
| concat_5L | lightgbm | +0.495 (n=20) |
| attention | ridge | +0.439 (n=20) |
| attention | mlp | +0.440 (n=20) |
| attention | lightgbm | +0.463 (n=20) |

### SaProt（已完成 20/20）
| 策略 | 读出 | mutant@modulo ρ |
|---|---|---|
| last_layer | ridge | +0.432 (n=20) |
| last_layer | mlp | +0.445 (n=20) |
| last_layer | lightgbm | +0.448 (n=20) |
| mean_20_33 | ridge | +0.433 (n=20) |
| mean_20_33 | mlp | +0.429 (n=20) |
| mean_20_33 | lightgbm | +0.439 (n=20) |
| concat_5L | ridge | +0.434 (n=20) |
| concat_5L | mlp | +0.406 (n=20) |
| concat_5L | lightgbm | +0.436 (n=20) |
| attention | ridge | +0.429 (n=20) |
| attention | mlp | +0.426 (n=20) |
| attention | lightgbm | +0.433 (n=20) |

---
生成: 2026-09-19 18:03:35
注: TOST 固定层选择=均值曲线最优层（m9 披露）；pooling n=20/模型为原投稿设计。