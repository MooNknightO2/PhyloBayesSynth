要实现基于上下文无关文法的系统发育学概率程序合成（通过贝叶斯方式）
也就是说，我们现在拿到发育树的结构和枝长数据，想要推断进化模型和参数。
1. 根据下面的描述定义上下文无关文法：
   N1:=(NoisyPhylo N2 N3)
   N2:=(Noise P)
   N3:=(CRB P)
       |(CRBD P P)
       |(TDB P P)
       |(TDBD P P P)
       |(PWBD P N3 N3)
       |(BAMM P N3 N3)
   其中N1到N3是非终结符，CRB, CRBD, TDB, TDBD, PWBD, BAMM是标识，分别指的是CRB恒定出生率，CRBD恒定出生死亡率，TDB由时间决定的出生率，TDBD由时间决定的出生死亡率，PWBD嵌套分段出生死亡率，BAMM是change-point（切换根据泊松过程发生，默认先验概率为0）。P指的是待填的参数。
   PWBD (Piecewise Birth-Death) 有一个参数 τ_frac∈(0,1) 表示断点在当前区间中的相对位置，以及两个子表达式 N3 分别描述近现世端与近根端的模型。嵌套 PWBD 可组合出任意多段、每段不同模型类型的分段BD过程。例如:
     (PWBD 0.5 (CRBD 0.3 0.1) (TDBD 0.5 0.2 0.1))  — 两段，前半CRBD后半TDBD
     (PWBD 0.4 (CRB 0.5) (PWBD 0.6 (CRBD 0.3 0.1) (TDB 0.8 -0.1)))  — 三段
   PWBD的似然使用通用Λ-参数化Riccati闭式解，对每段使用Λ=∫λ(s)ds、ε=μ/λ参数化，在断点处链式衔接E与D的初始条件，全程闭式O(n·K)计算。参考Stadler (2011)、Louca et al. (2022)分段BD模型理论。
2. 从先验中采样的函数Generate_Expression_From_Prior
   Expand $[(t_{ik}, \theta_1 \ldots \theta_{h_{ik}}, E_1 \ldots E_{n_{ik}})](N_i) := p_{ik} \prod_{j=1}^{n_{ik}} \gamma_{ik}(\theta_1, \ldots, \theta_{h_{ik}}) \prod_{j=1}^{n_{ik}} \text{Expand}[E_j](\tilde{N}_{ik}^j)$
   其中$t_{ik}$是某个标识，如TDB，CRB等，$\theta_1 \ldots \theta_{h_{ik}}$是该标识对应的参数，$E_1 \ldots E_{n_{ik}}$是该标识对应的表达式，$p_{ik}$是选择的概率，$\gamma_{ik}$是选择的参数的联合分布，$\tilde{N}_{ik}^j$是该标识对应的第j个非终结符。
   则有：$\text{Prior}[E] := \text{Expand}[E](N^{\text{start}})$
3. 实现Evaluate_Likelihood函数
   该函数输入一个表达式E和数据D，返回该表达式生成数据D的似然值。
   要求：根据表达式E的结构计算其生成数据D的似然值。
   其中，TDB模型没有灭绝，出生率的假设为$λ(t)=λ_0e^{z(t_0-t)}$,注意，这里只有λ_0和z是待确定的参数。TDBD模型有灭绝，为了简化，保持λ和μ的比值不变。
4. 实现语义的剪枝函数Sever：
    输入参数a表示在表达式E的解析树中的某个节点位置，返回该节点对应的非终结符Ni以及剪枝后的表达式，其中Esev在位置a处有一个空洞。
    然后实现转移算子，下面是伪代码：
    1: procedure GENERATE-NEW-EXPRESSION(O, E) ▷ observation $O \in \mathcal{X}$ and input expression $E \in \mathcal{L}$
    2: $a \sim \text{Uniform}(A_E)$ ▷ randomly select a node in parse tree
    3: $(N_i, E_{\text{sev}}) \leftarrow \text{Sever}_a[E]$ ▷ sever parse tree and return non-terminal symbol at sever point
    4: $E_{\text{sub}} \sim \text{Expand}[\cdot](N_i)$ ▷ generate random $E_{\text{sub}}$ with probability $\text{Expand}[E_{\text{sub}}](N_i)$
    5: $E' \leftarrow E_{\text{sev}}[E_{\text{sub}}]$ ▷ fill hole in $E_{\text{sev}}$ with expression $E_{\text{sub}}$
    6: $L \leftarrow \text{Likelihood}[E](O)$ ▷ evaluate likelihood for expression $E$ and observation $O$
    7: $L' \leftarrow \text{Likelihood}[E'](O)$ ▷ evaluate likelihood for expression $E'$ and observation $O$
    8: $p_{\text{accept}} \leftarrow \min\left\{1, (|A_E|/|A_{E'}|) \cdot (L'/L)\right\}$ ▷ compute the probability of accepting the mutation
    9: $r \sim \text{Uniform}([0, 1])$ ▷ draw a random number from the unit interval
    10: if $r < p_{\text{accept}}$ then ▷ if-branch has probability $p_{\text{accept}}$
    11: $\quad$ return $E'$ ▷ accept and return the mutated expression
    12: else ▷ else-branch has probability $1 - p_{\text{accept}}$
    13: $\quad$ return $E$ ▷ reject the mutated expression and return the input expression
5. 实现MCMC和SMC算法，伪代码分别如下：
MCMC:
Require: observation $O \in \mathcal{X}$, number of iterations $n \geq 1$  
1: procedure BAYESIAN-SYNTHESIS-MCMC($O, n$)  
2:     do  
3:         $E_0 \sim \mathrm{GENERATE-EXPRESSION-FROM-PRIO R}()$  $>$ generate $E_0$ with probability Prior $[E]$  
4:         while $\mathrm{EVALUATE-LIKELIHOOD}(O, E_0) = 0$  
5:         for $i = 1 \ldots n$  $>$ run $n$ sampling iterations  
6:             $E_i \sim \mathrm{GENERATE-NEW-EXPRESSION}(O, E_{i-1})$  
7: return $E_1, \ldots, E_n$
SMC:
Require: observations $(O_1, \ldots, O_J)$ where $O_i \in \mathcal{X}$; number of move iterations $n \geq 0$; number of particles $M \geq 1$; $\mathrm{ESS}_{\min} \in \{1, \ldots, M\}$
1: procedure Bayesian-Synthesis-SMC$(X, n, M)$  
2: $\quad$ for $\ell = 1 \ldots M$ do  
3: $\quad\quad E_0^\ell \sim \text{Generate-Expression-From-Prior}()$  
4: $\quad\quad w_0^\ell \leftarrow 1; \quad \hat{w}_{j-1}^\ell \leftarrow 1$  
5: $\quad$ for $j = 1 \ldots J$ do  
6: $\quad\quad // \text{reweight}$  
7: $\quad\quad$ for $\ell = 1 \ldots M$ do  
8: $\quad\quad\quad w_j^\ell \leftarrow \hat{w}_{j-1}^\ell \cdot \frac{\text{Evaluate-Likelihood}(O_j, E_{j-1}^\ell)}{\text{Evaluate-Likelihood}(O_{j-1}, E_{j-1}^\ell)}$  
9: $\quad\quad // \text{resample (adaptive)}$  
10: $\quad\quad$ if $\mathrm{ESS}(w_{j-1}^{1:M}) < \mathrm{ESS}_{\min}$ and $j < J$ then  
11: $\quad\quad\quad$ for $\ell = 1 \ldots M$ do  
12: $\quad\quad\quad\quad A_j^\ell \leftarrow \text{Categorical}(M; w_{j}^{1:M})$  
13: $\quad\quad\quad\quad \hat{w}_j^\ell \leftarrow 1$  
14: $\quad\quad$ else  
15: $\quad\quad\quad$ for $\ell = 1 \ldots M$ do  
16: $\quad\quad\quad\quad A_j^\ell \leftarrow \ell$  
17: $\quad\quad\quad\quad \hat{w}_j^\ell \leftarrow w_{j-1}^\ell$  
18: $\quad\quad // \text{rejuvenate}$  
19: $\quad\quad$ for $\ell = 1 \ldots M$ do  
20: $\quad\quad\quad E_j^\ell \leftarrow E_{j-1}^{A_j^\ell}$  
21: $\quad\quad\quad$ for $i = 1 \ldots n$ do  
22: $\quad\quad\quad\quad E_j^\ell \sim \text{Generate-New-Expression}(O_j, E_j^\ell)$  
23: return $E_{0:J}^{1:M}, w_{0:J}^{1:M}, A_{1:J}^{1:M}$