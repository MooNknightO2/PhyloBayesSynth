# 似然函数详解

## 通用记号

| 符号 | 含义 |
|------|------|
| $n$ | 叶节点（现存物种）数 |
| $T$ | 树高（根节点距今时间） |
| $\tau$ | 距今时间（present = 0，root = $T$） |
| $x_i$ | 第 $i$ 条枝的长度 |
| $\tau_c^{(i)},\;\tau_p^{(i)}$ | 第 $i$ 条枝的子/父节点距今时间 |
| $S = \sum_i x_i$ | 总枝长 |

树是重构树（只含现存物种），共 $2n-2$ 条枝、$n-1$ 个分支事件。

---

## 1. CRB — 恒定出生率（Yule 纯生模型）

**参数**: $\lambda$ （恒定出生率）

**物理图像**: 每条现存谱系以恒定速率 $\lambda$ 产生新分支，没有灭绝。整棵树就是一个纯生 Poisson 分支过程。

**推导思路**:
- 在任意时刻，有 $k$ 条谱系共存，总分支速率为 $k\lambda$
- 下一次分支事件的等待时间 $\sim \text{Exp}(k\lambda)$
- 选中特定谱系分叉的概率为 $1/k$
- 这两项消掉 $k$，每次事件贡献 $\lambda \cdot e^{-k\lambda \Delta t}$
- 对所有事件求积，$\sum k\Delta t$ 恰好等于总枝长 $S$

**公式**:

$$\log \mathcal{L}(\lambda) = \log(n-1)! + (n-1)\log\lambda - \lambda S$$

- $(n-1)!$：标记历史的排列数（拓扑常数）
- $\lambda^{n-1}$：$n-1$ 次分支事件各贡献一个 $\lambda$
- $e^{-\lambda S}$：所有枝上"未发生额外分支"的概率

---

## 2. CRBD — 恒定出生-死亡率

**参数**: $\lambda$（出生率），$\mu$（死亡率），要求 $\lambda > 0,\; \mu \geq 0$

**物理图像**: 每条谱系以速率 $\lambda$ 分支、速率 $\mu$ 灭绝。我们只观测到存活到现世的谱系（重构树），灭绝的谱系不可见。似然需要"整合掉"所有不可见的灭绝旁支。

**推导思路**:
- 对于每条枝，需要计算：该谱系在枝的时间段内（1）自身没有发生可见分支，（2）所有可能产生的隐形旁支最终都灭绝了
- 分母项 $(1 - \frac{\mu}{\lambda} e^{-(\lambda-\mu)\tau_c})^2$ 就是对"隐形旁支全部灭绝"的条件概率

**公式**:

$$\log \mathcal{L}(\lambda, \mu) = \log(n-1)! + (n-1)\log\lambda + \sum_{i=1}^{2n-2} \left[ -(\lambda-\mu)\,x_i - 2\log\!\left(1 - \frac{\mu}{\lambda}\,e^{-(\lambda-\mu)\,\tau_c^{(i)}}\right) \right]$$

各项含义：
- $e^{-(\lambda-\mu)x_i}$：枝 $i$ 上净存活（出生-死亡差速率）的概率
- $\frac{1}{(1 - \frac{\mu}{\lambda}\,e^{-(\lambda-\mu)\tau_c})^2}$：从子节点 $\tau_c$ 到现世 (0) 的存活条件因子。叶节点处 $\tau_c=0$，该项退化为 $(1-\mu/\lambda)^{-2}$

**退化为 CRB**: 当 $\mu = 0$ 时，分母 = 1，公式化简为 CRB。

---

## 3. TDB — 时间依赖出生率（无灭绝）

**参数**: $\lambda_0$（根部出生率），$z$（时间依赖系数）

**物理图像**: 出生率随时间变化，没有灭绝。

$$\lambda(\tau) = \lambda_0 \, e^{z(T - \tau)}$$

- $z > 0$：出生率从根到现世指数增长（加速多样化）
- $z < 0$：出生率从根到现世指数衰减（niche-filling，物种越多竞争越激烈）
- $z = 0$：退化为 CRB

**推导思路**:
- 与 CRB 类似，但每个分支事件的速率不再是常数 $\lambda$，而是事件发生时刻的 $\lambda(\tau_k)$
- 每条枝上"无分支事件"的概率不再是 $e^{-\lambda x_i}$，而是 $e^{-\int_{\tau_c}^{\tau_p} \lambda(\tau)\,d\tau}$

**定义累积速率**:

$$\Lambda(a, b) = \int_a^b \lambda(\tau)\,d\tau = \begin{cases} \dfrac{\lambda_0}{z}\left(e^{z(T-a)} - e^{z(T-b)}\right) & z \neq 0 \\[6pt] \lambda_0\,(b - a) & z = 0 \end{cases}$$

**公式**:

$$\log \mathcal{L}(\lambda_0, z) = \log(n-1)! + \sum_{k=1}^{n-1} \log\lambda(\tau_k) - \sum_{i=1}^{2n-2} \Lambda(\tau_c^{(i)},\, \tau_p^{(i)})$$

- $\sum \log\lambda(\tau_k)$：每个分支事件处的瞬时速率之积
- $\sum \Lambda$：所有枝上的累积速率之和（对应"无事件"概率）

---

## 4. TDBD — 时间依赖出生-死亡率（恒定 turnover）

**参数**: $\lambda_0$（根部出生率），$z$（时间依赖系数），$\varepsilon = \mu/\lambda$（turnover 比，$0 \leq \varepsilon < 1$）

**物理图像**: 出生率和死亡率都随时间变化，但它们的比值 $\varepsilon$ 保持恒定。

$$\lambda(\tau) = \lambda_0\,e^{z(T-\tau)}, \qquad \mu(\tau) = \varepsilon\,\lambda(\tau)$$

**推导思路**:
- CRBD 的时变推广：将 CRBD 公式中的常数速率替换为时变速率的积分
- 分子中的 $(\lambda-\mu)x_i$ 替换为 $(1-\varepsilon)\,\Lambda(\tau_c, \tau_p)$
- 分母中的 $e^{-(\lambda-\mu)\tau_c}$ 替换为 $e^{-(1-\varepsilon)\,\Lambda(0,\,\tau_c)}$

**公式**:

$$\log \mathcal{L}(\lambda_0, z, \varepsilon) = \log(n-1)! + \sum_{k=1}^{n-1} \log\lambda(\tau_k) + \sum_{i=1}^{2n-2} \left[ -(1-\varepsilon)\,\Lambda_i^{\text{branch}} - 2\log\!\left(1 - \varepsilon\,e^{-(1-\varepsilon)\,\Lambda_i^{\text{child}}}\right) \right]$$

其中：
- $\Lambda_i^{\text{branch}} = \Lambda(\tau_c^{(i)},\, \tau_p^{(i)})$：枝上的出生率积分
- $\Lambda_i^{\text{child}} = \Lambda(0,\, \tau_c^{(i)})$：从现世到子节点的出生率积分（存活条件）

**退化关系**:
- $\varepsilon = 0$ → TDB（无灭绝）
- $z = 0$ → CRBD（恒定速率）
- $\varepsilon = 0, z = 0$ → CRB

---

## 5. BAMM — 变点模型（泊松过程触发模型切换）

**参数**: $\eta$（shift 泊松速率），$M_{\text{bg}}$（背景多样化模型），$M_{\text{fg}}$（前景多样化模型）

**物理图像**: 树在 $M_{\text{bg}}$ 模型下演化；沿每条谱系，以泊松速率 $\eta$ 发生"regime shift"。一旦某条谱系发生 shift，该谱系及其所有后代切换到 $M_{\text{fg}}$ 模型。

**计算方法**: 自底向上递归遍历树，在每个节点同时追踪两种 regime 下的似然。

对于每个内部节点，每条子枝的贡献：

**fg regime**（已切换，不会再变）:

$$\ell_{\text{fg}} = f_{\text{fg}}(\tau_p, \tau_c) \cdot [\lambda_{\text{fg}}(\tau_c)]^{\mathbb{1}[\text{child 是内部节点}]} \cdot L_{\text{child}}^{\text{fg}}$$

**bg regime**（未切换，可能在此枝上发生 shift）:

$$\ell_{\text{bg}} = \underbrace{e^{-\eta\,\ell} \cdot f_{\text{bg}}(\tau_p, \tau_c) \cdot \lambda_{\text{bg}}(\tau_c) \cdot L_{\text{child}}^{\text{bg}}}_{\text{此枝无 shift}} + \underbrace{\int_{\tau_c}^{\tau_p} \eta\,e^{-\eta(\tau_p - s)} \cdot f_{\text{bg}}(\tau_p, s) \cdot f_{\text{fg}}(s, \tau_c) \cdot \lambda_{\text{fg}}(\tau_c) \cdot L_{\text{child}}^{\text{fg}} \, ds}_{\text{此枝在 $s$ 处发生 shift}}$$

其中：
- $f_M(\tau_p, \tau_c)$ 是模型 $M$ 的单枝因子（即上面 CRB/CRBD/TDB/TDBD 公式中每条枝的贡献）
- $\ell = \tau_p - \tau_c$ 是枝长
- 积分用 **Gauss-Legendre 求积** 数值计算（12 点）

**最终似然**:

$$\log \mathcal{L} = \log(n-1)! + \log\lambda_{\text{bg}}(T) + \sum_{\text{root 的子枝}} \log\ell_{\text{bg}}$$

树从根节点开始处于 bg regime，自底向上汇总。

---

## 观测噪声（污染混合模型）

所有模型的似然最终经过观测噪声参数 $\sigma$ 调节。采用污染混合（contamination mixture）框架：

$$\mathcal{L}_{\text{final}} = (1 - \pi)\,\mathcal{L}_{\text{model}} + \pi\,\mathcal{L}_{\text{noise}}, \qquad \pi = \frac{\sigma}{1 + \sigma}$$

其中 $\mathcal{L}_{\text{noise}}$ 是噪声基线分布——对 CRB 模型的 $\lambda$ 取 Exponential 先验后的边际似然：

$$\mathcal{L}_{\text{noise}} = \frac{\beta \,[\Gamma(n)]^2}{(S + \beta)^n}, \qquad \beta = \frac{1}{\text{scale}_\lambda}$$

- $\sigma = 0$（$\pi = 0$）：完全信任结构化模型 $\mathcal{L}_{\text{model}}$
- $\sigma \to \infty$（$\pi \to 1$）：完全退化为噪声基线
- 中间值：以概率 $1-\pi$ 认为数据来自结构化进化模型，以概率 $\pi$ 认为来自非结构化噪声

---

## 各模型关系总览

```
              z=0          ε=0          加 shift
CRB ──────► CRBD        CRB ──────► TDB        任意模型
 │           │            │           │         ──────► BAMM(η, bg, fg)
 │  加μ      │  加μ       │  加z      │  加z
 ▼           ▼            ▼           ▼
CRBD ─────► ???         TDB ──────► TDBD
        (时变CRBD)
```

| 模型 | 参数数 | 灭绝 | 时变 | 变点 |
|------|--------|------|------|------|
| CRB  | 1      | ✗    | ✗    | ✗    |
| CRBD | 2      | ✓    | ✗    | ✗    |
| TDB  | 2      | ✗    | ✓    | ✗    |
| TDBD | 3      | ✓    | ✓    | ✗    |
| BAMM | 1+递归 | 取决于子模型 | 取决于子模型 | ✓ |

