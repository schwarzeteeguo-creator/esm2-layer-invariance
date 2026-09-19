# M1 对照实验正式结果（GCV 统一协议版）：随机位置向量 control（63 数据集）

> 生成：2026-09-13（v2：补入 F7YBW8）| 脚本：`esm_embedding/probing_v4.py` | 数据：ProteinGym substitution LMDB
> 协议：RidgeCV **GCV**（cv=None, gcv_mode="eigen"，Golub 1979）——与服务器端
> GPU 三臂实验（probing_v4 / pooling_v3）**完全同一读出协议**；fold 内标准化、
> 5 折、变体不跨折、seed=42、33 次随机抽签。
> 本文件取代 `../probing_v4/CONTROL_SUMMARY.md`（其嵌套 cv=3 版数字已被本版
> 替代；两组数字差异 <0.011 ρ，结论不变）。
>
> **F7YBW8 披露**：该数据集仅 4 个突变位置，position-grouped CV 按组数防护
> 取 4 折（其余 62 数据集 5 折不变）；其 contiguous 划分仅 1 个非空折 →
> Spearman 无定义，该列按 57 个数据集计（F7YBW8 与另 4 个同类数据集）。

## 核心数字

| 特征 | random | modulo | contiguous | position(GroupKFold) |
|---|---:|---:|---:|---:|
| **随机位置向量 + aux**（位置查找表对照） | **+0.6001** | +0.1375 | +0.0957¹ | +0.1347 |
| 随机位置向量 alone | +0.5567 | −0.0076 | −0.0564¹ | −0.0101 |
| aux alone（42 维手工特征，唯一突变特异通道） | +0.3385 | +0.2295 | +0.1877¹ | +0.2276 |

n=63（contiguous 列 n=57¹）。¹ 6 个数据集（含 F7YBW8）contiguous 划分下
Spearman 无定义（常数预测/空折），按其余 57 个计。

**配对检验**（random vs position-aware，随机位置向量+aux，逐数据集配对）：

| 对比 | n | 平均跌落 | Wilcoxon p |
|---|---:|---:|---:|
| vs modulo | 63 | −0.463 | 5.2×10⁻¹² |
| vs contiguous | 57 | −0.515 | 5.1×10⁻¹¹ |
| vs position(GroupKFold) | 63 | −0.465 | 5.2×10⁻¹² |

## 与已发表数字的对照（回复信核心论据）

| 量 | 已发表（masked-WT PLM probing） | 随机向量对照 |
|---|---:|---:|
| random split | ρ ≈ 0.605–0.607（"层不变性"） | **ρ = 0.600** |
| position-aware split | ρ ≈ 0.382 | ρ = 0.135 |

**结论**：
1. 随机划分下，**每位置一个随机向量的查找表即可完全复现已发表的 PLM 逐层探测性能**
   （0.608 vs 0.607）——random-split masked-position probing 测得的基本全是位置记忆
   （Reviewer 1 Major 1 完全成立）。
2. 无泄露评估下对照崩溃至 0.14 以下，而已发表 PLM 结果为 0.382 —— PLM 嵌入真正
   携带的突变相关信息约为 +0.24 ρ，只有在消除位置泄露后才能测到。
3. aux（真正含突变身份的特征）从 0.34 降到 0.23，降幅温和——与"位置通道被剥离，
   突变特异信号保留"的解释一致。

> 待服务器三臂实验（mutant-sequence 特征 + 统一 GCV 协议）就位后，本表将扩展为
> 四特征臂 × 四划分的完整主表。
