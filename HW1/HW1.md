# 支持向量機 (SVM) 與多層感知器 (MLP) 訓練過程完整推導

本文詳細推導 SVM 和 MLP 的訓練過程，重點說明**為什麼**要用這些方法，以及如何實現權重調整 $\mathbf{W}^* = \mathbf{W} + \Delta\mathbf{W}$。

---

# 第一部分：支持向量機 (SVM) 訓練過程

## 一、問題設定與幾何直觀

### 1.1 基本設定

考慮二維空間中的線性分類問題：

- **資料點**：$\mathbf{x} = (x_1, x_2)^T$
- **權重向量**：$\mathbf{w} = (w_1, w_2)^T$
- **決策邊界**：$\mathbf{w}^T\mathbf{x} + b = 0$
- **類別標籤**：$y \in \\{-1, +1\\}$

### 1.2 Margin 的幾何推導

**定義兩條平行邊界線：**

- 正類邊界：$\mathbf{w}^T\mathbf{x} + b = +1$
- 負類邊界：$\mathbf{w}^T\mathbf{x} + b = -1$

**在兩邊界上任取兩點 $\mathbf{x}_m$ 和 $\mathbf{x}_n$：**

$$
\begin{cases}
w_1 x_{1m} + w_2 x_{2m} + b = +1 \\
w_1 x_{1n} + w_2 x_{2n} + b = -1
\end{cases}
$$

**相減消去 $b$：**

$$
w_1(x_{1m} - x_{1n}) + w_2(x_{2m} - x_{2n}) = 2
$$

向量形式：

$$
\mathbf{w}^T(\mathbf{x}_m - \mathbf{x}_n) = 2
$$

**計算兩邊界間的垂直距離：**

向量 $\mathbf{w}$ 是決策邊界的**法向量** (normal vector)。兩點連線 $(\mathbf{x}_m - \mathbf{x}_n)$ 在法向量 $\mathbf{w}$ 上的投影長度即為 margin：

$$
L = \frac{\mathbf{w}^T(\mathbf{x}_m - \mathbf{x}_n)}{\|\mathbf{w}\|} = \frac{2}{\|\mathbf{w}\|}
$$

> **幾何意義**：
> 
> $$\|\mathbf{x}_m - \mathbf{x}_n\| \cdot \cos\theta = L$$
> 
> 其中 $\cos\theta = \frac{\mathbf{w}^T(\mathbf{x}_m - \mathbf{x}_n)}{\|\mathbf{w}\| \cdot \|\mathbf{x}_m - \mathbf{x}_n\|}$

**結論：Margin 公式**

$$
\boxed{L = \frac{2}{\|\mathbf{w}\|}}
$$

若邊界設為 $\mathbf{w}^T\mathbf{x} + b = \pm k$，則：

$$
L = \frac{2k}{\|\mathbf{w}\|}
$$

---

## 二、最佳化問題的建立

### 2.1 為什麼要最大化 Margin？

**統計學習理論告訴我們**：

- Margin 越大 → 模型的**泛化能力**越好
- 可以容忍更多的測試資料變異
- 降低 overfitting 風險

**目標**：最大化 margin

$$
\max_{\mathbf{w}, b} L = \max_{\mathbf{w}, b} \frac{2}{\|\mathbf{w}\|}
$$

### 2.2 為什麼轉換成最小化 $\|\mathbf{w}\|^2$？

**數學技巧 1：去除常數**

常數 2 不影響最優解的位置，可以忽略：

$$
\max \frac{2}{\|\mathbf{w}\|} \Longleftrightarrow \max \frac{1}{\|\mathbf{w}\|}
$$

**數學技巧 2：最大化倒數 = 最小化原式**

因為 $\frac{1}{\|\mathbf{w}\|}$ 與 $\|\mathbf{w}\|$ 單調遞減：

$$
\max \frac{1}{\|\mathbf{w}\|} \Longleftrightarrow \min \|\mathbf{w}\|
$$

**數學技巧 3：平方化**

為什麼要平方？

1. **便於求導**：$\|\mathbf{w}\| = \sqrt{w_1^2 + w_2^2}$ 有根號，求導複雜
2. **單調性不變**：在 $\|\mathbf{w}\| > 0$ 時，$f(x) = x^2$ 單調遞增
3. **數學性質好**：二次函數平滑，優化理論成熟

$$
\min \|\mathbf{w}\| \Longleftrightarrow \min \|\mathbf{w}\|^2 \Longleftrightarrow \min \frac{1}{2}\|\mathbf{w}\|^2
$$

前面加 $\frac{1}{2}$ 是為了求導後消掉係數 2。

### 2.3 完整的原始問題 (Primal Problem)

$$
\begin{aligned}
\min_{\mathbf{w}, b} \quad & \frac{1}{2}\|\mathbf{w}\|^2 \\
\text{s.t.} \quad & y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, N
\end{aligned}
$$

**約束條件的意義**：

- $y_i = +1$ 時：$\mathbf{w}^T\mathbf{x}_i + b \geq +1$ （正類在正邊界外側）
- $y_i = -1$ 時：$\mathbf{w}^T\mathbf{x}_i + b \leq -1$ （負類在負邊界外側）

---

## 三、為什麼用 Lagrange 對偶？

### 3.1 原始問題的困難

**問題類型**：帶不等式約束的最佳化問題

如果沒有約束，直接對 $\mathbf{w}, b$ 求導即可。但現在有 $N$ 個不等式約束：

$$
g_i(\mathbf{w}, b) = 1 - y_i(\mathbf{w}^T\mathbf{x}_i + b) \leq 0, \quad i = 1, \ldots, N
$$

**為什麼不能直接求導？**

- 不等式約束無法直接代入目標函數
- 約束可能在最優點處**不活躍** (inactive)
- 需要判斷哪些約束是 binding（等號成立）

### 3.2 Lagrange 乘子法的引入

**核心思想**：將約束條件轉化為懲罰項

$$
\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = f(\mathbf{w}, b) + \sum_{i=1}^{N} \alpha_i g_i(\mathbf{w}, b)
$$

其中：
- $f(\mathbf{w}, b) = \frac{1}{2}\|\mathbf{w}\|^2$ 是目標函數
- $g_i(\mathbf{w}, b) = 1 - y_i(\mathbf{w}^T\mathbf{x}_i + b)$ 是約束
- $\alpha_i \geq 0$ 是 Lagrange multipliers

**SVM 的 Lagrangian：**

$$
\boxed{\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^{N} \alpha_i[y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1]}
$$

**為什麼是減號？**

因為約束是 $1 - y_i(\mathbf{w}^T\mathbf{x}_i + b) \leq 0$，重寫為 $y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1 \geq 0$，所以用減號。

### 3.3 為什麼要用 KKT 條件？

**KKT (Karush-Kuhn-Tucker) 條件** 是含不等式約束最佳化問題的必要條件。

**KKT 條件包含：**

1. **Stationarity（穩定性）**：
   
   $$\nabla_{\mathbf{w}} \mathcal{L} = 0, \quad \frac{\partial \mathcal{L}}{\partial b} = 0$$

2. **Primal feasibility（原始可行性）**：
   
   $$y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$$

3. **Dual feasibility（對偶可行性）**：
   
   $$\alpha_i \geq 0$$

4. **Complementary slackness（互補鬆弛）**：
   
   $$\alpha_i[y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1] = 0$$

**互補鬆弛的意義**：

- 若 $\alpha_i > 0$，則 $y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$ → 該點是 **Support Vector**
- 若 $y_i(\mathbf{w}^T\mathbf{x}_i + b) > 1$，則 $\alpha_i = 0$ → 該點不影響決策邊界

---

## 四、對偶問題推導

### 4.1 求解 Stationarity 條件

**對 $\mathbf{w}$ 做偏微分：**

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \mathbf{w} - \sum_{i=1}^{N} \alpha_i y_i \mathbf{x}_i = 0
$$

$$
\Rightarrow \quad \boxed{\mathbf{w}^* = \sum_{i=1}^{N} \alpha_i y_i \mathbf{x}_i} \quad \text{（關鍵公式！）}
$$

**這個公式說明**：
- 最佳權重 $\mathbf{w}^*$ 是訓練樣本的**線性組合**
- 只有 $\alpha_i > 0$ 的樣本（Support Vectors）有貢獻
- 這就是為什麼叫「支持向量」機！

**對 $b$ 做偏微分：**

$$
\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^{N} \alpha_i y_i = 0
$$

$$
\Rightarrow \quad \boxed{\sum_{i=1}^{N} \alpha_i y_i = 0}
$$

### 4.2 代回得到對偶問題

將 $\mathbf{w}^* = \sum \alpha_i y_i \mathbf{x}_i$ 代入 $\mathcal{L}$：

$$
\begin{aligned}
\mathcal{L}(\boldsymbol{\alpha}) 
&= \frac{1}{2}\left\|\sum_{i=1}^{N} \alpha_i y_i \mathbf{x}_i\right\|^2 - \sum_{i=1}^{N} \alpha_i\left[y_i\left(\sum_{j=1}^{N} \alpha_j y_j \mathbf{x}_j^T\mathbf{x}_i + b\right) - 1\right]
\end{aligned}
$$

**展開第一項：**

$$
\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (\mathbf{x}_i^T\mathbf{x}_j)
$$

**展開第二項並利用 $\sum \alpha_i y_i = 0$ 消去 $b$：**

$$
-\sum_{i=1}^{N}\sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (\mathbf{x}_i^T\mathbf{x}_j) + \sum_{i=1}^{N} \alpha_i
$$

**合併得到對偶問題：**

$$
\boxed{
\begin{aligned}
\max_{\boldsymbol{\alpha}} \quad & W(\boldsymbol{\alpha}) = \sum_{i=1}^{N} \alpha_i - \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (\mathbf{x}_i^T\mathbf{x}_j) \\
\text{s.t.} \quad & \sum_{i=1}^{N} \alpha_i y_i = 0 \\
& \alpha_i \geq 0, \quad i = 1, \ldots, N
\end{aligned}
}
$$

---

## 五、為什麼是二次規劃 (QP) 問題？

### 5.1 QP 問題的定義

**標準形式：**

$$
\begin{aligned}
\min_{\mathbf{x}} \quad & \frac{1}{2}\mathbf{x}^T Q \mathbf{x} + \mathbf{p}^T\mathbf{x} \\
\text{s.t.} \quad & A\mathbf{x} = \mathbf{b} \\
& \mathbf{x} \geq 0
\end{aligned}
$$

**SVM 對偶問題的特點：**

1. **目標函數是二次的**：含有 $\alpha_i \alpha_j$ 項
2. **約束是線性的**：$\sum \alpha_i y_i = 0$ 和 $\alpha_i \geq 0$
3. **目標函數是凸的**：$Q$ 矩陣半正定（因為是 Gram matrix）

### 5.2 為什麼 QP 很重要？

**理論保證：**

1. **凸最佳化**：局部最優 = 全域最優
2. **唯一解**：在嚴格凸的情況下
3. **成熟算法**：有高效的求解器

**SVM 的 Q 矩陣：**

$$
Q_{ij} = y_i y_j (\mathbf{x}_i^T\mathbf{x}_j)
$$

這是一個 **Gram matrix**，半正定，保證了凸性。

---

## 六、為什麼用 SMO 算法？

### 6.1 直接求解 QP 的問題

**標準 QP solver 的問題：**

- 時間複雜度：$O(N^3)$（$N$ 是樣本數）
- 空間複雜度：$O(N^2)$（需要存儲整個 Q 矩陣）
- 當 $N$ 很大時（如 10,000+ 樣本），幾乎不可行

### 6.2 SMO 的核心思想

**Sequential Minimal Optimization (SMO)**：

**思想**：不要一次優化所有 $\alpha_i$，而是每次只優化**兩個** $\alpha_i, \alpha_j$。

**為什麼是兩個？**

因為約束 $\sum \alpha_i y_i = 0$，如果只優化一個 $\alpha_i$，它的值被其他變數固定了。至少需要兩個變數才能在滿足約束下進行優化。

**SMO 演算法步驟：**

```
1. 初始化所有 αi = 0
2. Repeat until convergence:
   a. 選擇兩個 αi, αj（違反 KKT 最嚴重的）
   b. 固定其他所有 αk (k ≠ i, j)
   c. 在約束下解析求解 αi, αj 的最優值
   d. 更新 αi, αj
3. Return α*
```

**為什麼能解析求解？**

當只有兩個變數時，問題變成：

$$
\begin{aligned}
\max \quad & L(\alpha_i, \alpha_j) \\
\text{s.t.} \quad & \alpha_i y_i + \alpha_j y_j = C \text{ (常數)} \\
& 0 \leq \alpha_i, \alpha_j \leq C
\end{aligned}
$$

這是一個**單變數**二次規劃（因為 $\alpha_j$ 可用 $\alpha_i$ 表示），有**解析解**！

### 6.3 SMO 的優勢

1. **不需要存儲 Q 矩陣**：只計算需要的元素
2. **時間複雜度**：實際約 $O(N^2)$ 到 $O(N^{2.3})$
3. **記憶體效率**：$O(N)$ 而非 $O(N^2)$
4. **簡單實作**：每個子問題都有閉式解

---

## 七、權重的計算：為什麼不是 $\mathbf{w} = \mathbf{w} + \Delta\mathbf{w}$？

### 7.1 SVM 的權重更新方式

**關鍵理解**：SVM 的訓練過程是：

```
1. 迭代更新 α (使用 SMO)
   α₁ ← α₁ + Δα₁
   α₂ ← α₂ + Δα₂
   ...
   (重複直到收斂)

2. 一次性計算 w*
   w* = Σ αᵢ* yᵢ xᵢ  ← 只在最後執行一次
```

**對比 MLP：**

```
1. 每次迭代都更新 W
   W ← W + ΔW    ← 每個 epoch 都執行
   (重複直到收斂)
```

### 7.2 為什麼 SVM 不直接更新 $\mathbf{w}$？

**數學原因：**

1. **對偶問題更簡單**：
   - Primal: 變數 = $d + 1$（$w$ 的維度 + $b$）+ $N$ 個約束
   - Dual: 變數 = $N$（$\alpha$ 的個數），但只有 1 個等式約束

2. **Kernel trick**：
   - 對偶問題中只出現 $\mathbf{x}_i^T\mathbf{x}_j$（內積）
   - 可以替換成 kernel function $K(\mathbf{x}_i, \mathbf{x}_j)$
   - 實現**非線性分類**而無需顯式計算高維 $\mathbf{w}$

3. **稀疏性**：
   - 大部分 $\alpha_i = 0$
   - 只有 Support Vectors ($\alpha_i > 0$) 對 $\mathbf{w}^*$ 有貢獻
   - 典型情況：只有 10-30% 的樣本是 SV

### 7.3 計算 $\mathbf{w}^*$ 和 $b^*$

**計算 $\mathbf{w}^*$：**

$$
\mathbf{w}^* = \sum_{i \in SV} \alpha_i^* y_i \mathbf{x}_i
$$

只對 Support Vectors 求和，非常高效！

**計算 $b^*$：**

選擇任一 Support Vector $\mathbf{x}_k$（滿足 $0 < \alpha_k^* < C$），利用 $y_k(\mathbf{w}^{*T}\mathbf{x}_k + b^*) = 1$：

$$
b^* = y_k - \mathbf{w}^{*T}\mathbf{x}_k
$$

實務上對所有 SV 求平均更穩定：

$$
b^* = \frac{1}{|SV|}\sum_{k \in SV}\left(y_k - \mathbf{w}^{*T}\mathbf{x}_k\right)
$$

---

## 八、完整訓練流程總結

```
【SVM 訓練流程】

輸入：訓練資料 {(xᵢ, yᵢ)}ᵢ₌₁ᴺ

步驟 1: 初始化
  α ← 0

步驟 2: 使用 SMO 迭代更新 α
  Repeat:
    選擇違反 KKT 最嚴重的 αᵢ, αⱼ
    解析求解並更新 αᵢ, αⱼ
  Until convergence (滿足 KKT 條件)

步驟 3: 計算 w*（只執行一次）
  w* ← Σ αᵢ* yᵢ xᵢ  (只對 Support Vectors 求和)

步驟 4: 計算 b*
  b* ← 平均值 of (yₖ - w*ᵀxₖ) over Support Vectors

輸出：決策函數 f(x) = sign(w*ᵀx + b*)
```

**關鍵點**：
- ✅ 更新的是 $\alpha$，不是 $\mathbf{w}$
- ✅ $\mathbf{w}^*$ 在最後一次性計算
- ✅ 不使用 $\mathbf{w} = \mathbf{w} + \Delta\mathbf{w}$ 的形式

---

# 第二部分：多層感知器 (MLP) 訓練過程

## 一、前向傳播 (Forward Propagation)

### 1.1 網路結構

考慮 $L$ 層神經網路：

- **輸入層**：$\mathbf{a}^{(0)} = \mathbf{x} \in \mathbb{R}^{d}$
- **隱藏層** $l = 1, \ldots, L-1$
- **輸出層**：$l = L$

### 1.2 第 $l$ 層的計算

**線性組合 (Affine transformation)：**

$$
\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}
$$

其中：
- $\mathbf{W}^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$：權重矩陣
- $\mathbf{b}^{(l)} \in \mathbb{R}^{n_l}$：bias 向量
- $n_l$：第 $l$ 層的神經元數量

**非線性激活 (Activation)：**

$$
\mathbf{a}^{(l)} = f(\mathbf{z}^{(l)})
$$

常用激活函數：
- Sigmoid: $\sigma(z) = \frac{1}{1+e^{-z}}$
- ReLU: $\text{ReLU}(z) = \max(0, z)$
- Tanh: $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$

---

## 二、損失函數與梯度下降

### 2.1 為什麼需要損失函數？

**目的**：量化模型預測與真實值的差距

**常用損失函數：**

**1. 均方誤差 (MSE) - 用於回歸：**

$$
J = \frac{1}{2}\|\hat{\mathbf{y}} - \mathbf{y}\|^2 = \frac{1}{2}\sum_{k=1}^{n_L}(\hat{y}_k - y_k)^2
$$

**2. 交叉熵 (Cross-Entropy) - 用於分類：**

$$
J = -\sum_{k=1}^{n_L} y_k \log(\hat{y}_k)
$$

**整個訓練集的損失：**

$$
L = \frac{1}{N}\sum_{n=1}^{N} J^{(n)} + \frac{\lambda}{2}\sum_{l=1}^{L}\|\mathbf{W}^{(l)}\|_F^2
$$

第二項是 **L2 regularization**，防止 overfitting。

### 2.2 為什麼用梯度下降？

**問題**：最小化 $L(\mathbf{W}, \mathbf{b})$

**特點**：
- **非凸函數**：存在多個局部最小值
- **高維空間**：參數數量可達百萬級
- **無解析解**：無法直接求導等於零

**梯度下降的思想**：

沿著**負梯度方向**移動，逐步降低損失：

$$
\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \frac{\partial L}{\partial \mathbf{W}^{(l)}}
$$

**這就是 $\mathbf{W} = \mathbf{W} + \Delta\mathbf{W}$ 的實現！**

其中：

$$
\Delta\mathbf{W}^{(l)} = -\eta \frac{\partial L}{\partial \mathbf{W}^{(l)}}
$$

---

## 三、為什麼需要反向傳播 (Backpropagation)？

### 3.1 計算梯度的挑戰

**問題**：如何高效計算 $\frac{\partial L}{\partial \mathbf{W}^{(l)}}$ ？

**直接計算的困難：**

對於第 1 層的權重，損失函數的計算路徑是：

$$
\mathbf{W}^{(1)} \to \mathbf{z}^{(1)} \to \mathbf{a}^{(1)} \to \mathbf{z}^{(2)} \to \cdots \to \mathbf{a}^{(L)} \to L
$$

需要多次應用 chain rule，計算複雜度極高！

### 3.2 Backpropagation 的思想

**核心**：利用 chain rule 從後往前**重複使用**中間結果

**定義誤差項：**

$$
\boldsymbol{\delta}^{(l)} = \frac{\partial L}{\partial \mathbf{z}^{(l)}}
$$

這是損失對第 $l$ 層**線性輸出**的偏微分。

**為什麼定義這個？**

1. 便於遞推：$\boldsymbol{\delta}^{(l)}$ 可以從 $\boldsymbol{\delta}^{(l+1)}$ 計算
2. 計算權重梯度時會用到

---

## 四、Backpropagation 詳細推導

### 4.1 輸出層誤差

**對於輸出層 $L$：**

$$
\boldsymbol{\delta}^{(L)} = \frac{\partial L}{\partial \mathbf{z}^{(L)}} = \frac{\partial L}{\partial \mathbf{a}^{(L)}} \odot f'(\mathbf{z}^{(L)})
$$

**Chain rule 展開：**

$$
\frac{\partial L}{\partial z_i^{(L)}} = \sum_k \frac{\partial L}{\partial a_k^{(L)}} \cdot \frac{\partial a_k^{(L)}}{\partial z_i^{(L)}}
$$

由於 $a_i^{(L)} = f(z_i^{(L)})$，所以：

$$
\frac{\partial a_i^{(L)}}{\partial z_i^{(L)}} = f'(z_i^{(L)})
$$

**對於 MSE 損失：**

$$
L = \frac{1}{2}\sum_k (\hat{y}_k - y_k)^2 = \frac{1}{2}\sum_k (a_k^{(L)} - y_k)^2
$$

$$
\frac{\partial L}{\partial a_k^{(L)}} = a_k^{(L)} - y_k = \hat{y}_k - y_k
$$

如果輸出層用 identity activation ($f(z) = z$, $f'(z) = 1$)：

$$
\boxed{\boldsymbol{\delta}^{(L)} = \hat{\mathbf{y}} - \mathbf{y}}
$$

**對於 Cross-Entropy + Softmax：**

$$
L = -\sum_k y_k \log(\hat{y}_k)
$$

$$
\hat{y}_k = \frac{e^{z_k^{(L)}}}{\sum_j e^{z_j^{(L)}}}
$$

可以證明：

$$
\boxed{\boldsymbol{\delta}^{(L)} = \hat{\mathbf{y}} - \mathbf{y}}
$$

（Softmax 的導數恰好抵消！）

### 4.2 隱藏層誤差的遞推

**對於隱藏層 $l < L$：**

使用 chain rule：

$$
\frac{\partial L}{\partial z_i^{(l)}} = \sum_j \frac{\partial L}{\partial z_j^{(l+1)}} \cdot \frac{\partial z_j^{(l+1)}}{\partial a_i^{(l)}} \cdot \frac{\partial a_i^{(l)}}{\partial z_i^{(l)}}
$$

**第二項：**

$$
z_j^{(l+1)} = \sum_k W_{jk}^{(l+1)} a_k^{(l)} + b_j^{(l+1)}
$$

$$
\frac{\partial z_j^{(l+1)}}{\partial a_i^{(l)}} = W_{ji}^{(l+1)}
$$

**第三項：**

$$
\frac{\partial a_i^{(l)}}{\partial z_i^{(l)}} = f'(z_i^{(l)})
$$

**合併得到向量形式：**

$$
\boxed{\boldsymbol{\delta}^{(l)} = \left[(\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)}\right] \odot f'(\mathbf{z}^{(l)})}
$$

其中 $\odot$ 是 element-wise 相乘 (Hadamard product)。

**這個公式的意義：**

- $(\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)}$：將下一層的誤差**反向傳播**回來
- $f'(\mathbf{z}^{(l)})$：乘以當前層激活函數的導數

### 4.3 權重和 Bias 的梯度

**權重梯度推導：**

$$
\frac{\partial L}{\partial W_{ij}^{(l)}} = \frac{\partial L}{\partial z_i^{(l)}} \cdot \frac{\partial z_i^{(l)}}{\partial W_{ij}^{(l)}}
$$

因為 $z_i^{(l)} = \sum_k W_{ik}^{(l)} a_k^{(l-1)} + b_i^{(l)}$：

$$
\frac{\partial z_i^{(l)}}{\partial W_{ij}^{(l)}} = a_j^{(l-1)}
$$

所以：

$$
\frac{\partial L}{\partial W_{ij}^{(l)}} = \delta_i^{(l)} \cdot a_j^{(l-1)}
$$

**矩陣形式：**

$$
\boxed{\frac{\partial L}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T}
$$

**Bias 梯度：**

$$
\frac{\partial L}{\partial b_i^{(l)}} = \frac{\partial L}{\partial z_i^{(l)}} \cdot \frac{\partial z_i^{(l)}}{\partial b_i^{(l)}} = \delta_i^{(l)} \cdot 1
$$

$$
\boxed{\frac{\partial L}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}}
$$

---

## 五、權重更新：實現 $\mathbf{W}^* = \mathbf{W} + \Delta\mathbf{W}$

### 5.1 標準梯度下降

$$
\mathbf{W}^{(l)}_{\text{new}} = \mathbf{W}^{(l)}_{\text{old}} - \eta \frac{\partial L}{\partial \mathbf{W}^{(l)}}
$$

**定義權重調整量：**

$$
\Delta\mathbf{W}^{(l)} = -\eta \frac{\partial L}{\partial \mathbf{W}^{(l)}} = -\eta \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T
$$

**更新規則等價於：**

$$
\boxed{\mathbf{W}^{(l)} = \mathbf{W}^{(l)} + \Delta\mathbf{W}^{(l)}}
$$

**注意**：$\Delta\mathbf{W}^{(l)}$ 是**負的**（因為有負號），代表沿著梯度**下降**方向。

### 5.2 為什麼 MLP 使用這種更新方式？

**與 SVM 的對比：**

| 特性 | SVM | MLP |
|------|-----|-----|
| 最佳化性質 | Convex (凸) | Non-convex (非凸) |
| 是否有解析解 | 有（透過對偶問題） | 無 |
| 更新方式 | 更新 $\alpha$，最後算 $\mathbf{w}^*$ | 直接更新 $\mathbf{W}$ |
| 更新頻率 | $\alpha$ 每次迭代更新，$\mathbf{w}$ 只算一次 | $\mathbf{W}$ 每次迭代都更新 |
| 是否用梯度 | 否（用 KKT 條件） | 是（必須用梯度） |

**MLP 為什麼必須用 $\mathbf{W} = \mathbf{W} + \Delta\mathbf{W}$：**

1. **非凸性**：沒有全域最優解的解析式
2. **高維複雜**：無法像 SVM 轉換到對偶空間
3. **逐步逼近**：只能透過梯度資訊一步步改進

### 5.3 不同的梯度下降變體

**1. Batch Gradient Descent (BGD)：**

每次用**全部**訓練資料計算梯度：

$$
\Delta\mathbf{W}^{(l)} = -\eta \frac{1}{N}\sum_{n=1}^{N} \frac{\partial J^{(n)}}{\partial \mathbf{W}^{(l)}}
$$

- 優點：穩定，收斂方向準確
- 缺點：慢，記憶體需求大

**2. Stochastic Gradient Descent (SGD)：**

每次只用**一個**樣本：

$$
\Delta\mathbf{W}^{(l)} = -\eta \frac{\partial J^{(n)}}{\partial \mathbf{W}^{(l)}}
$$

- 優點：快，可以線上學習
- 缺點：震盪大，不穩定

**3. Mini-batch Gradient Descent：**

每次用**一小批** (如 32, 64, 128) 樣本：

$$
\Delta\mathbf{W}^{(l)} = -\eta \frac{1}{|B|}\sum_{n \in B} \frac{\partial J^{(n)}}{\partial \mathbf{W}^{(l)}}
$$

- 平衡了速度和穩定性
- **現代深度學習的標準做法**

### 5.4 進階 Optimizer

**1. Momentum：**

加入動量項，加速收斂：

$$
\begin{aligned}
\mathbf{v}_t &= \beta \mathbf{v}_{t-1} + \eta \nabla L \\
\Delta\mathbf{W}^{(l)} &= -\mathbf{v}_t
\end{aligned}
$$

**2. Adam (Adaptive Moment Estimation)：**

結合動量和自適應學習率：

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2 \\
\Delta\mathbf{W}^{(l)} &= -\eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
\end{aligned}
$$

其中 $\mathbf{g}_t = \nabla L$。

**所有這些 Optimizer 的共同點**：都是 $\mathbf{W} = \mathbf{W} + \Delta\mathbf{W}$ 的形式！

---

## 六、SVM vs MLP 完整對比

### 6.1 數學性質對比

| 項目 | SVM | MLP |
|------|-----|-----|
| **問題類型** | Convex QP | Non-convex optimization |
| **目標函數** | $\min \frac{1}{2}\|\mathbf{w}\|^2$ | $\min L(\mathbf{W}, \mathbf{b})$ |
| **約束** | 線性不等式 | 無（但有 regularization） |
| **最優性** | Global optimum | Local optimum |
| **解的唯一性** | 唯一（在嚴格凸情況下） | 不唯一（多個局部最優） |

### 6.2 演算法對比

| 項目 | SVM | MLP |
|------|-----|-----|
| **核心演算法** | SMO | Backpropagation + GD |
| **更新對象** | Dual variables ($\alpha$) | Primal variables ($\mathbf{W}$) |
| **更新公式** | $\alpha_i \leftarrow \alpha_i + \Delta\alpha_i$ | $\mathbf{W} \leftarrow \mathbf{W} + \Delta\mathbf{W}$ |
| **需要梯度？** | 否 | 是 |
| **權重計算** | $\mathbf{w}^* = \sum \alpha_i^* y_i \mathbf{x}_i$ | 直接最佳化 $\mathbf{W}$ |
| **計算次數** | $\mathbf{w}$ 只算一次 | $\mathbf{W}$ 每次迭代都更新 |

### 6.3 權重調整方式的本質差異

**SVM：**
```
訓練過程：
  更新 α₁, α₂, ..., αₙ  (多次迭代)
  ↓
最後計算 w*:
  w* = Σ αᵢ* yᵢ xᵢ  (執行一次)
  
特點：
✗ 不使用 w = w + Δw
✓ 間接更新（透過 α）
✓ 最終解是訓練樣本的線性組合
```

**MLP：**
```
訓練過程：
  每個 epoch:
    W ← W + ΔW  (直接更新)
    
特點：
✓ 使用 W = W + ΔW
✓ 直接更新權重
✓ 需要多次迭代逐步逼近
```

### 6.4 為什麼有這些差異？

**根本原因：問題性質不同**

**SVM (Convex QP)：**
- 有強理論保證（KKT 條件）
- 可以轉換到對偶空間求解
- 解具有稀疏性（只有 SV 有貢獻）
- 可以用 kernel trick 處理非線性

**MLP (Non-convex)：**
- 沒有解析解
- 必須用梯度資訊
- 需要多次迭代
- 可以近似任意函數（Universal Approximation Theorem）

---

## 七、實務建議

### 7.1 何時用 SVM？

✅ **適合的情況：**
- 資料維度高但樣本數少（$d > N$）
- 需要理論保證（convex optimization）
- 想要稀疏解（只用部分樣本）
- 資料可能線性可分或用 kernel 可分

❌ **不適合的情況：**
- 樣本數超大（>100,000）
- 需要多層非線性轉換
- 需要 end-to-end 學習（如影像、語音）

### 7.2 何時用 MLP？

✅ **適合的情況：**
- 大量訓練資料
- 複雜的非線性關係
- 需要多層特徵抽取
- 可以用 GPU 加速

❌ **不適合的情況：**
- 訓練資料很少（容易 overfit）
- 需要理論保證
- 資源有限（訓練時間長）

---

## 八、Python 簡單實作對比

### 8.1 SVM 實作

```python
from sklearn.svm import SVC
import numpy as np

# 訓練 SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)

# 取得結果
w_star = svm.coef_[0]      # 直接得到 w*
b_star = svm.intercept_[0]
alpha = svm.dual_coef_[0]  # dual variables
sv = svm.support_vectors_

# 驗證: w* = Σ αᵢyᵢxᵢ
w_manual = np.sum(alpha[:, np.newaxis] * sv, axis=0)
print(f"w* = {w_star}")
print(f"驗證: {w_manual}")
print(f"Support Vectors: {len(sv)} / {len(X_train)}")
```

### 8.2 MLP 實作

```python
import numpy as np

class SimpleMLP:
    def __init__(self, sizes):
        # 初始化權重
        self.W = [np.random.randn(y, x) * 0.01 
                  for x, y in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros((y, 1)) for y in sizes[1:]]
    
    def forward(self, x):
        a = x
        self.activations = [a]
        self.zs = []
        
        for W, b in zip(self.W, self.b):
            z = W @ a + b
            a = sigmoid(z)
            self.zs.append(z)
            self.activations.append(a)
        return a
    
    def backward(self, x, y, lr):
        # Backpropagation
        delta = (self.activations[-1] - y) * sigmoid_prime(self.zs[-1])
        
        deltas = [delta]
        for l in range(len(self.W)-1, 0, -1):
            delta = (self.W[l].T @ delta) * sigmoid_prime(self.zs[l-1])
            deltas.insert(0, delta)
        
        # 更新權重: W = W + ΔW
        for l in range(len(self.W)):
            grad_W = deltas[l] @ self.activations[l].T
            grad_b = deltas[l]
            
            delta_W = -lr * grad_W  # 計算 ΔW
            delta_b = -lr * grad_b
            
            self.W[l] += delta_W  # 關鍵：W = W + ΔW
            self.b[l] += delta_b

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

# 使用
mlp = SimpleMLP([2, 4, 1])
for epoch in range(1000):
    for x, y in zip(X_train, y_train):
        mlp.forward(x)
        mlp.backward(x, y, lr=0.1)  # 每次都更新 W
```

---

## 九、總結

### 核心差異

**SVM 的權重獲得方式：**
```
1. 求解對偶問題 → 得到 α*
2. 計算 w* = Σ αᵢ* yᵢ xᵢ  (一次性)
3. 不使用 w = w + Δw
```

**MLP 的權重獲得方式：**
```
1. 初始化 W
2. 重複：
   - Forward propagation
   - Backpropagation 計算梯度
   - W = W + ΔW  (每次迭代)
3. 直到收斂
```

### 關鍵公式對比

| | SVM | MLP |
|---|-----|-----|
| **最佳化目標** | $\min \frac{1}{2}\|\mathbf{w}\|^2$ | $\min L(\mathbf{W})$ |
| **更新公式** | $\alpha \leftarrow \alpha + \Delta\alpha$ | $\mathbf{W} \leftarrow \mathbf{W} + \Delta\mathbf{W}$ |
| **權重公式** | $\mathbf{w}^* = \sum \alpha_i^* y_i \mathbf{x}_i$ | $\mathbf{W}^* = \mathbf{W} - \eta\nabla L$ |
| **使用 $\Delta\mathbf{w}$** | ✗ | ✓ |

---

**License:** MIT  
**最後更新：** 2025-10-30
