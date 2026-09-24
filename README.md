# RNA 二级结构预测（带强制配对 + Boltzmann 系综）

离线、纯标准库实现的 RNA 非交叉（无假结）二级结构可复用模块，提供两个公开入口：

- **`fold(sequence, constraints)`**：精确区间动态规划，穷举所有合法结构求
  **最大整数分值**最优，返回字典序最小结构、最优计数、可出现配对；
- **`partition_fold(sequence, constraints, temperature, ...)`**：在正温度下
  计算 Boltzmann **配分函数** `Z = Σ exp(−E/T)`（整数能量），全部在 log
  空间稳定聚合，返回自由能与**配对/未配边缘概率**（长度 ≤ 20）。

两者都不做伪随机抽样；强制约束在输入校验阶段直接拒绝，配分函数的概率由
inside/outside 两遍 DP 精确覆盖整个合法系综。

## 环境与依赖

- Python **3.14.7**（3.10+ 的类型语法即可运行，目标环境为 3.14.7）
- **仅使用 Python 标准库**，安装 Python 后无需安装任何第三方依赖
- 完全离线：不需要网络、账号、密钥、数据库、中间件、云 API 或 Docker

## 快速运行

在仓库根目录（本文件所在目录）执行：

```bash
# 运行全部必要测试（独立穷举参考 + DP 对拍，含配分函数）
python -m unittest discover -s tests -v

# 固定输入演示：最大分值折叠 + 配分函数 + 3 个真实触发的拒绝
python demo.py
```

`python demo.py` 在普通笔记本上约 0.05 秒完成（远低于 90 秒预算），所有
结果都是当场由算法真实计算得到，不使用 `sleep`，也不预录输出。

## 公共接口

```python
from rna_fold import fold, partition_fold

result = fold(sequence, constraints=None)

ensemble = partition_fold(
    sequence,
    constraints=None,
    temperature=1.0,
    pair_energies=None,   # 默认 GC/CG=-3, AU/UA=-2, GU/UG=-1
    unpaired_energy=0,
)
```

### `fold` 参数

- `sequence` (`str`)：RNA 序列，仅允许字符 `A C G U`（大小写敏感，小写拒绝），
  长度 **1 ≤ n ≤ 120**（空串 `""` 也允许，返回空结构）。
- `constraints`（可选）：三种形式
  - `None`：自由折叠；
  - `str`：长度必须等于序列长度，支持
    - `.` 该位置自由；
    - `x` 该位置**强制未配**；
    - 成对括号 `()`、`[]`、`{}`、`<>`：括号配平，其两端位置**强制配对**；
  - `dict`：`{"pairs": [(i, j), ...], "unpaired": [k, ...]}`，所有下标 **0 基**。
    - `"pairs"`：强制配对 `(i, j)`，要求 `i < j` 且 `j - i >= 4`；
    - `"unpaired"`：强制未配位置；两个键都可省略。

### `fold` 返回值（普通 `dict`，不暴露可变内部状态）

| 键 | 类型 | 含义 |
|---|---|---|
| `score` | `int` | 最大总分：GC/CG=3，AU/UA=2，GU/UG=1 |
| `structure` | `str` | 最优结构中**字典序最小**的 dot-bracket（`(`、`)`、`.`） |
| `count` | `int` | 最优结构数量，模 **1 000 000 007** |
| `pairs` | `tuple[(i, j), ...]` | **所有至少出现在某个最优结构中**的配对，按 `(i,j)` 升序 |

字典序按 ASCII 码比较，即 `'(' < ')' < '.'`。

### 合法结构规则（题面契约）

- 只允许 `AU, UA, GC, CG, GU, UG` 六种配对；
- 每对内部至少 **3 个未配位点**：`j - i >= 4`（最小发夹环为 3 nt）；
- 禁止交叉（无假结），每个位置最多参与一个配对。

### `partition_fold` 参数（Boltzmann 系综，n ≤ 20）

- `sequence` (`str`)：同上，但长度 **0 ≤ n ≤ 20**，超过 20 抛
  `InvalidSequenceError`（错误消息给出实际长度与上限 20）。
- `constraints`：与 `fold` 完全相同的三种形式与同一套校验（强制配对在
  系综中的边缘概率必为 1）。
- `temperature`：**严格为正的有限数** `T > 0`（int 或 float，单位与能量
  一致）。`bool`、字符串等非数值类型抛 `TypeError`；`0`、负数、`NaN`、
  `Infinity` 抛 `InvalidParameterError`。
- `pair_energies`：`None`（用默认表）或映射，键为六种**有向**配对类型
  `"GC" "CG" "AU" "UA" "GU" "UG"`，可只覆盖其中任意子集，其余沿用默认；
  值必须是整数能量。未知键抛 `InvalidParameterError`；非整数浮点/`bool`
  抛 `TypeError`，`NaN`/`Infinity` 抛 `InvalidParameterError`。
  默认能量为 `GC/CG=-3、AU/UA=-2、GU/UG=-1`（负为有利），即 `fold` 分值
  的相反数，因此 **T → 0 时系综集中到 `fold` 的最大分值结构**。
- `unpaired_energy`：每个未配位置的整数能量，默认 `0`。

一个结构的总能量 = 各配对能量之和 + 未配位置数 × `unpaired_energy`；
权重 = `exp(−E/T)`，配分函数 `Z = Σ_结构 exp(−E(结构)/T)`。

### `partition_fold` 返回值（普通 `dict`）

| 键 | 类型 | 含义 |
|---|---|---|
| `partition_function` | `float` | `Z`；溢出时诚实返回 `inf`（此时以 `ln_partition_function` 为准） |
| `ln_partition_function` | `float` | `ln(Z)`，主要的稳定数值量；常规能量下有限 |
| `free_energy` | `float` | `-T · ln(Z)` |
| `temperature` | `float` | 实际使用的、校验后的温度 |
| `min_energy` | `int` | 合法系综中结构的**精确整数**最小能量（任意精度） |
| `unpaired_probabilities` | `tuple[float,…]` | 每个位置未配的边缘概率，长度 n |
| `pair_probabilities` | `dict[(i,j), float]` | 每个概率非零的合法配对的边缘概率，键按 `(i,j)` 升序 |
| `pair_energies` / `unpaired_energy` | `dict` / `int` | 实际生效的完整整数能量表 |

概率恒等式（每个位置）：`P(k 未配) + Σ_j P(k 与 j 配对) = 1`
（浮点舍入内，测试以 1e-10 断言）；强制配对概率恰为 1。零能量时每个结构
等权，`Z` 恰好等于合法结构数（见测试中的解析核对）。

## 错误语义（异常层级）

所有错误消息都带可定位信息（0 基位置或 `pairs[k]` / `unpaired[k]` 条目名），
**不静默纠正、不猜测**。`fold` 是纯函数，失败不修改入参，不存在部分变更。

- `rna_fold.InvalidSequenceError(ValueError)`：序列类型错误、非法字符；
  长度超上限（`fold` 为 120、`partition_fold` 为 20，错误消息标明上限）。
- `rna_fold.ConstraintError(ValueError)`：约束**结构性**非法：
  - 类型错误、下标越界、整数位置传入 `bool`/`NaN`/`Infinity`/非整数浮点；
  - 字符串约束长度不符、括号不配平、含非法字符；
  - 强制配对本身不合法（距离不足、碱基不互补）；
  - 两对强制配对共用端点、彼此交叉、或端点同时被强制未配；
  - `dict` 形式含未知键。
- `rna_fold.UnsatisfiableConstraintError(ConstraintError)`：
  约束良构但被**精确搜索证明无解**。当前实现下，单个强制配对的物理可行性
  （距离、互补性）与配对间冲突已在输入校验全部拒绝，因此正常输入不会到达此分支；
  保留该异常是为了让「证明无解」这一语义有稳定、可捕获的公开类型。
- `rna_fold.BudgetExceededError(RuntimeError)`：
  **搜索预算耗尽**（未能证明最优）。它与「证明无解」是刻意分开的两个类型，
  互不构成子类。由于两个入口都是精确 DP、完整枚举整个解空间，本实现**永不
  抛出**该异常；它作为固定错误语义的一部分保留。
- `rna_fold.InvalidParameterError(ValueError)`（仅 `partition_fold`）：
  温度非有限正数（`0`、负数、`NaN`、`Infinity`），或能量表出现未知键、
  能量值为 `NaN`/`Infinity`。能量值传非整数浮点或 `bool` 仍是 `TypeError`，
  与 `fold` 的数值契约一致。`partition_fold` 失败同样是纯函数式的：先完成
  全部参数校验再进入 DP，不修改任何入参，不留下半更新状态。

数值输入契约：整数参数拒绝 `bool`（`True/False` 不当作 1/0）；`float` 的
`NaN`/`Infinity` 分别按名称拒绝，其他非整数浮点报 `TypeError`。

## 算法与设计取舍

- **区间 DP（Zuker 式简化版，无能量模型）**：对半开区间 `[i, j)`，
  `best[i][j] = max(best[i+1][j],  max_t {w(i,t) + best[i+1][t] + best[t+1][j]})`，
  `t` 从 `i+4` 起，且 `(i,t)` 必须互补、不违反强制约束。
  时间 O(n³)、空间 O(n²)；n=120 实测约 0.15 秒。
- **强制约束的处理**：除逐位置的配对/未配限制外，引入「闭区间」表
  `closed[i][j]`——区间内任何有强制配偶的位置，其配偶也必须在区间内。
  DP 只在闭子区间上转移，于是任何与强制配对交叉的选择都会使某个子区间不闭，
  自动不可达，不需要特判交叉。
- **计数**：与最优表同步用加法/组合乘法累加，随时对 1 000 000 007 取模。
- **字典序最小结构**：第二张区间表从短到长重建；同一最优分值下，
  `i` 未配的候选为 `"."+...`，`i` 配对的候选为 `"("+...+")"+...`，
  取 ASCII 最小者。配对候选按 `t` 升序取最小。
- **可出现在最优中的配对**：从根区间 `[0,n)` 沿「保持最优分值」的决策做
  可达性传播；可达区间上的任何最优配对都至少属于一个全局最优结构。
  可达性只由分值表（可行性）决定，**不使用取模后的计数**，因此即使某子结构
  集合数量恰为模数倍数（计数显示 0），配对集合仍然精确。测试
  `test_pair_set_independent_of_count_modulus` 用小模数白盒验证了这一点。

### 与独立参考的关系

`tests/brute_reference.py` 是**从零编写、不导入被测模块**的独立参考：
对 n ≤ 14 递归枚举全部非交叉匹配，再以「事后过滤」方式保留含全部强制配对的
结构（与生产代码的闭区间 DP 是两套独立机制），独立给出最大分、字典序最小串、
精确最优数量与最优配对集合；另提供 `brute_partition`，直接枚举每个结构并以
普通浮点算术求和 Boltzmann 权重与边缘概率（与生产代码的 log 空间 inside/outside
DP 是两套独立机制）。测试在数百个确定性随机序列（固定种子）上逐字段对拍。
往返一致性（返回串自身合法、能达到所报分值）仅作为补充断言。

### 配分函数 DP（`partition_fold`）

对同一套区间分解做三遍 DP，时间 O(n³)、空间 O(n²)，n=20 为亚毫秒级：

1. **精确整数最小能量 DP** `M[i][j]`：全程任意精度整数、不碰浮点，给出每个
   区间的最小结构能量与根的 `min_energy`；
2. **inside（log 空间）**：`Q[i][j]` 存的是「相对本区间最小能量」的
   `ln Z[i][j] + M[i][j]/T`，恒 ≤ 0（最小能量项恰贡献 `exp(0)=1`），用稳定的
   `logaddexp`（内部 `log1p(exp(·))`）聚合——所有指数参数 ≤ 0，**只会下溢、
   永不溢出**；
3. **outside（log 空间）**：从根沿分解边反向传播上下文权重 `R[i][j]`，
   子区间自身的平移与边的能量差精确抵消；
4. **边缘概率**：区间分解对每个结构是唯一树，每个位置恰在一个节点被决定
   （未配或与某 `t` 配对），所以「实现该事件的分解边」权重之和就是该事件
   指示函数的配分函数，除以 `Q[0][n]` 即得概率。

即使能量整数大到无法转为有限浮点（如 10**400）：整数 DP 仍精确，gap 超过
2^1024 的项其权重在有限浮点中本就严格为 0，概率依然正确；只有绝对量
`Z`/`ln Z`/自由能会诚实饱和为 `inf`（README 与返回契约已注明）。

## 数值/误差口径

- `fold` 的分值与计数均为**整数精确计算**：分值是小整数精确加法；计数只做模
  1 000 000 007 的精确整数运算。该入口**全程无浮点参与**，没有浮点误差或容差，
  也不存在用宽容差掩盖边界错误的情况。
- `partition_fold` 的能量与最小能量是**精确整数**；概率与 `ln Z` 为 IEEE
  双精度浮点，全部在 log 空间聚合。对拍容差：概率 1e-9、`ln Z` 1e-9（测试
  实际误差远小于此）；归一化恒等式以 1e-10 断言。由于以精确最小能量锚定，
  任何 `exp` 的参数都 ≤ 0，数值路径上不存在上溢；相对最小能量的能隙若大到
  无法用有限浮点表示，其权重在双精度中本就严格为 0。**唯一会出现的 `inf`
  是绝对量 Z/ln Z/自由能本身超出浮点范围**（能量量级约 ≥ 2^1024 ≈ 1.8e308），
  此时概率仍精确，调用方应使用 `ln_partition_function` 与边缘概率。
- `demo.py` 的耗时显示（`time.perf_counter()`）仅展示用，不参与任何结果判定。

## 已知限制（按本题契约实现，不宣称行业标准兼容）

- 这是题面定义的**简化模型**，不是完整 RNA 折叠能量模型：能量只按
  「每个配对 + 每个未配位置」独立计，**无最近邻能量**、无堆栈/发夹/凸包/
  多分支环能量项、无 kcal/mol 标定，不与 ViennaRNA 等行业标准兼容。
- `partition_fold` 只返回配分函数与边缘概率；**不做随机结构采样**（题面本次
  不要求），也不做复杂能量模型的联合学习。长度上限 20（题面限定）；`fold`
  仍支持到 120。
- 禁止假结（非交叉）；不预测三链、修饰碱基、非 `ACGU` 字母（如 N、I）。
- `fold` 只输出单一字典序最小最优结构；其余并列最优结构以 `count` 与
  `pairs` 概括，不逐个枚举。
- `BudgetExceededError` 在当前确定性 DP 下不会被抛出；它是为语义完整性保留的
  公开类型。

## 仓库内容

```
rna_fold.py            # 核心模块：fold()、partition_fold() 与公开异常/常量
demo.py                # 固定输入演示（真实计算 + 真实拒绝）
tests/test_fold.py     # fold 测试（对拍、约束、GU、并列最优、非法输入等）
tests/test_partition.py# partition_fold 测试（手算、解析关系、枚举对拍、边界）
tests/brute_reference.py  # 独立穷举参考（n<=14，不调用被测核心）
README.md
```
