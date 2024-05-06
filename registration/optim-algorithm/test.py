import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
import sys

# 设置终端的编码为UTF-8
sys.stdout.reconfigure(encoding='utf-8')

original_string = """
CMA-ES（Covariance Matrix Adaptation Evolution Strategy）算法通过精心设计的更新规则确保协方差矩阵 \(C\) 保持正定和对称。在进化策略中，维持协方差矩阵的正定性是至关重要的，因为这保证了搜索椭球体的形状合理，且可以生成有效的新样本。下面详细解释 CMA-ES 算法中相关的机制。

### 对称性维护
1. **初始设置**：
   - 在 CMA-ES 中，协方差矩阵 \(C\) 通常初始化为单位矩阵 \(I\)，显然是对称的。

2. **更新规则**：
   - 在每一代中，协方差矩阵 \(C\) 的更新遵循特定的对称保持公式。最常见的更新形式是：
     \[
     C \leftarrow (1 - c_{\text{cov}}) C + c_{\text{cov}} \frac{1}{\mu_{\text{cov}}} \sum_{i=1}^{\mu} w_i (y_i y_i^T)
     \]
     这里，\(y_i\) 是基于父代最优解重采样得到的差向量，\(w_i\) 是权重（通常基于排名），\(c_{\text{cov}}\) 是协方差学习率，\(\mu_{\text{cov}}\) 是有效选择集大小。由于每项 \(y_i y_i^T\) 都是对称的，整个更新项也是对称的。

### 正定性维护
1. **正定的起始点**：
   - 如前所述，协方差矩阵起始于单位矩阵 \(I\)，它是正定的。

2. **正定性的递归保持**：
   - 由于每个 \(y_i y_i^T\) 都是一个秩一的正定矩阵（外积形成的矩阵），因此在协方差矩阵的更新中添加这样的矩阵仍然保持协方差矩阵的正定性。
   - 更新公式中的加权平均也是一个关键因素。只要 \(C\) 在更新之前是正定的，加权 \(y_i y_i^T\) 的和（作为一个正定矩阵的加权和）也是正定的，因此 \(C\) 在更新后也保持正定。

3. **数值稳定性**：
   - 在实际实现中，可能会加入一小量的正值（如 \(\epsilon I\)，其中 \(\epsilon\) 是一个小常数）到协方差矩阵中，以保证其数值稳定性和正定性。这种技术称为正则化，可以防止因样本数量不足或其他数值问题导致的协方差矩阵退化。

### 结论
CMA-ES 算法通过精心设计的更新规则和初值设置确保协方差矩阵 \(C\) 始终是对称和正定的。这些性质对于算法的有效运行至关重要，因为它们保证了生成的多变量正态分布是合理的，且能有效地探索目标函数的搜索空间。维持这些性质的机制不仅确保了算法的理论正确性，也保证了其实际应用中的稳定性和效率。
"""
terms_to_remove = ["\( ", " \)", "\[\n", "\n\]", "\[ \n", " \n\]", "\n/\n", " $"]
terms_to_fill = ["$", "$", "$", "$", "$", "$", "\n", "$"]
# 循环替换多个子串
count = 0
for term in terms_to_remove:
    original_string = original_string.replace(term, terms_to_fill[count])
    count+=1

file_path = "output.txt"

with open(file_path, 'w', encoding='utf-8') as file:
    file.write(original_string)