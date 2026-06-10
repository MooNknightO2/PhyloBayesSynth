## 1 实验结果

按照上次讨论结果，以及Feras Saad的邮件建议，增加了SMC的合成过程。
在对CRBD/TDBD的合成数据恢复中，均能成功且接近原表达式。

但在BAMM上还是遇到问题：
真实模型：`(BAMM 0.15 (CRBD 0.4 0.05) (CRBD 1.0 0.65))`
参数：`--J 20 --M 100 --n_move 15`
SMC了五轮之后给出的仍是TDBD，且BAMM在链中占比较少（虽然BAMM拟合能力较强，但多参数导致其随机化产生的表达式似然太低了）
所以怀疑是否是BAMM的先验概率太少了？

另外，我找了多个不同的表达式，期望原表达式似然能高过MLE的TDBD和CRBD，但没有成功。

（这里一个可能原因是：CRBD/TDBD 经过了在所有树上的MLE，而 BAMM 使用的是**固定的生成参数 θ_true**。）
但问题在于：如果也上MLE的话，BAMM的似然会更高，更能拟合数据，但依旧不是原表达式。

| 模型 | 参数 | 累积 log-L | Δ vs BAMM_true |
|------|------|-----------|----------------|
| **BAMM 真实参数** | η=0.15, bg=(0.4, 0.05), fg=(1.0, 0.65) | 2877.70 | 0 |
| CRB 全局 MLE | λ=0.5212 | 2776.25 | −101.45 |
| CRBD 全局 MLE | λ=0.9554, μ=0.7283 | 2889.03 | **+11.33** |
| TDBD 全局 MLE | λ₀=2.1168, z=−0.075, ε=0.8807 | 2899.14 | **+21.44** |

## 2 学界关于 BAMM 可辨识性的争论

我在尝试解决这个问题的时候询问大模型，得到了关于可辨识性的争论资料。但有些涉及生物学背景，我还没有完全理解：

本实验观察到的「BAMM 与 TDBD 难以区分」现象并非偶然——它根植于出生-死亡模型的数学结构，是系统发育学中一个被广泛讨论的已知问题。

### 2.1 Louca & Pennell (2020)：同余类定理

Louca & Pennell 在 *Nature* 上证明了一个根本性的不可辨识性结果：

> 对于任意给定的现存物种时间树和任意候选多样化历史，存在**无穷多个**替代多样化历史与之具有完全相同的似然。

> **参考文献**：Louca, S. & Pennell, M.W. (2020). Extant timetrees are consistent with a myriad of diversification histories. *Nature*, 580, 502–505.

### 2.2 Moore et al. (2016) vs Rabosky (2017)：BAMM 的似然函数之争

Moore, Höhna, Cettineo, Davis & Huelsenbeck (2016) 在 *PNAS* 上对 BAMM 提出了三项批评：

1. **似然函数不正确**：BAMM 未考虑不可观测的速率变化事件（发生在已灭绝谱系上的 regime shift），导致似然函数是近似的而非精确的。
2. **后验对先验极度敏感**：速率变化个数的后验分布对 Poisson 过程的先验参数 $\eta$ 高度敏感，即使数据量很大也无法克服。
3. **速率估计不可靠**：在模拟数据上，BAMM 的参数估计表现较差。

Rabosky (2017) 在 *Systematic Biology* 上逐一反驳：

- Moore 等人提出的替代似然函数本身有数学缺陷（概率可超出 [0,1]）；
- 先验敏感性在合理的先验设置下并不严重；
- 在速率变化确实可被检测到的模拟树上，BAMM 表现良好，而表现不佳的树本身在统计上与恒定速率模型不可区分。

> **参考文献**：
>
> Moore, B.R. et al. (2016). Critically evaluating the theory and performance of Bayesian analysis of macroevolutionary mixtures. *PNAS*, 113(34), 9569–9574.
>
> Rabosky, D.L. (2017). Is BAMM flawed? Theoretical and practical concerns in the analysis of multi-rate diversification models. *Systematic Biology*, 67(3), 477–498.

### 2.3 Meyer & Wiens (2018)：高阶分类单元的速率估计

Meyer & Wiens (2018) 从另一个角度加深了对 BAMM 可辨识性的质疑。他们发现在高阶分类单元（如科、目级别）的系统发育中，BAMM 给出的速率变化数目和位置在不同的采样密度和先验设置下差异很大，且 BAMM 倾向于将速率异质性归因于个别分支而非整体趋势。

这为本实验提供了旁证：当多样化速率的变化可以被解读为"沿特定谱系的跳变"（BAMM）或"全局的平滑趋势"（TDBD）时，有限的树形数据通常无法区分两种解读。

> **参考文献**：Meyer, A.L.S. & Wiens, J.J. (2018). Estimating diversification rates for higher taxa: BAMM can give problematic estimates of rates and rate shifts. *Evolution*, 72(1), 39–53.

### 2.4 Morlon (2014)：时变模型的统一框架

Morlon (2014) 在综述中指出，时间依赖的出生-死亡模型（如 TDBD）和 regime-shift 模型（如 BAMM）是描述多样化速率变异的两大范式。从统计角度看，两者可以被视为对同一连续体的不同离散化：

- **TDBD**：速率是时间的光滑函数，用参数化曲线（指数、多项式等）描述全局趋势。
- **BAMM**：速率是分段常数（或分段参数化），由 Poisson 过程的 change-point 决定跳变位置。

当真实的生物过程不是严格的"突变式切换"时，两种范式产生的树形分布会高度重叠。

> **参考文献**：Morlon, H. (2014). Phylogenetic approaches for studying diversification. *Ecology Letters*, 17(4), 508–525.

### 2.5 与本实验的联系

| 文献结论 | 本实验验证 |
|---------|-----------|
| 同余类定理：存在无穷多模型具有相同似然 | TDBD MLE 的累积似然比 BAMM θ_true 高 +21 nats |
| BAMM 的信号在数据不足时难以检测 | 即使用 20 棵 n=80 的同源树联合推断，SMC 仍收敛到 TDBD |
| regime-shift 模型与时变模型可互相近似 | BAMM 边际化后的有效速率 ≈ TDBD 的指数函数形式 |
| 模型选择依赖参数优化公平性 | 固定 θ_true 的 BAMM 必然输给 MLE 优化的 TDBD/CRBD |

### 2.6 可能的方向

- **增加 BAMM 先验概率**：让 SMC 更容易采样到 BAMM 粒子。但是否会导致简单模型（如 CRB）被 BAMM 过度拟合？
- **替换或扩展 DSL 中的 BAMM 模型**：换用bayesian_inference_phylogenetics论文中提到的ClaDS系列模型或分段常数 BD 模型，看看是否能提高可辨识性。
