# 支持向量機 (Support Vector Machine, SVM) 訓練過程與最佳權重推導

本文詳細推導了線性可分支持向量機的訓練過程,核心目標是找到最大化分類間隔的超平面參數 $\mathbf{w}$ 和 $b$。

## 一、SVM 最優化問題的定義

SVM 的訓練目標是尋找一個超平面 $\mathbf{w}^T \mathbf{x} + b = 0$,使得訓練集中的所有點都能被正確分類,且**間隔 (Margin)** 達到最大。

### 1.1 約束條件

對於所有訓練樣本 $(\mathbf{x}_i, y_i)$,必須滿足函數間隔大於等於 1 的約束:

$$y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, N$$

其中 $y_i \in \{-1, +1\}$ 是類別標籤。

### 1.2 間隔最大化

幾何間隔 (Geometric Margin) 為 $\gamma = \dfrac{1}{\|\mathbf{w}\|}$。最大化 $\gamma$ 等價於最小化 $\|\mathbf{w}\|$。

**原始最優化問題 (Primal Problem):**

$$
\begin{aligned}
&\min_{\mathbf{w}, b} \quad \frac{1}{2} \|\mathbf{w}\|^2 \\
&\text{s.t.} \quad y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, N
\end{aligned}
$$

---

## 二、拉格朗日對偶問題推導

為了解決帶約束的優化問題,我們引入拉格朗日乘子 $\alpha_i \geq 0$。

### 2.1 構造拉格朗日函數

$$L(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2} \|\mathbf{w}\|^2 - \sum_{i=1}^{N} \alpha_i [y_i (\mathbf{w}^T \mathbf{x}_i + b) - 1]$$

### 2.2 求解最優條件 (KKT 條件)

對 $L$ 關於 $\mathbf{w}$ 和 $b$ 求偏導數並令其為零:

**1. 對 $\mathbf{w}$ 求偏微分:**

$$\frac{\partial L}{\partial \mathbf{w}} = \mathbf{w} - \sum_{i=1}^{N} \alpha_i y_i \mathbf{x}_i = 0$$

$$\Rightarrow \quad \mathbf{w}^* = \sum_{i=1}^{N} \alpha_i y_i \mathbf{x}_i$$

**2. 對 $b$ 求偏微分:**

$$\frac{\partial L}{\partial b} = - \sum_{i=1}^{N} \alpha_i y_i = 0$$

$$\Rightarrow \quad \sum_{i=1}^{N} \alpha_i y_i = 0$$

### 2.3 對偶問題 (Dual Problem)

將上述條件代回拉格朗日函數,得到只關於 $\boldsymbol{\alpha}$ 的對偶問題:

$$
\begin{aligned}
&\max_{\boldsymbol{\alpha}} \quad \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (\mathbf{x}_i^T \mathbf{x}_j) \\
&\text{s.t.} \quad \sum_{i=1}^{N} \alpha_i y_i = 0, \quad \alpha_i \geq 0, \quad i = 1, \ldots, N
\end{aligned}
$$

---

## 三、最佳權重 $\mathbf{w}^*$ 的實現

對偶問題是一個**二次規劃 (Quadratic Programming, QP)** 問題,通常使用 **SMO (Sequential Minimal Optimization)** 算法來高效求解。

### 3.1 求解 $\boldsymbol{\alpha}^*$

SMO 算法每次選擇兩個 $\alpha_i, \alpha_j$ 進行優化,將大規模 QP 問題分解為一系列可解析求解的子問題。訓練過程不斷迭代,直到 $\boldsymbol{\alpha}$ 滿足 KKT 條件。

### 3.2 最佳權重向量 $\mathbf{w}^*$ 的確定

當 $\boldsymbol{\alpha}^*$ 被求出後,最佳權重向量可直接計算:

$$\mathbf{w}^* = \sum_{i=1}^{N} \alpha_i^* y_i \mathbf{x}_i$$

**權重調整的完成方式:**

與 MLP 的梯度下降不同,SVM 的訓練**不是**透過 $\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} + \Delta \mathbf{w}$ 逐步迭代,而是通過求解對偶問題**直接**獲得 $\mathbf{w}^*$。只有 $\alpha_i^* > 0$ 的樣本 (即**支持向量**) 對 $\mathbf{w}^*$ 有貢獻。

### 3.3 最佳偏置 $b^*$ 的確定

選取任意一個支持向量 $\mathbf{x}_k$ (滿足 $\alpha_k^* > 0$),可解出:

$$b^* = y_k - \mathbf{w}^{*T} \mathbf{x}_k$$

實際應用中常對所有支持向量求平均以提高穩定性。

---

# 多層感知器 (MLP) 訓練過程: 反向傳播推導與權重調整

本文詳細推導使用反向傳播算法 (Backpropagation) 訓練 MLP 的過程,重點闡述如何計算梯度以及實現權重的迭代調整 $\mathbf{W}^* = \mathbf{W} + \Delta \mathbf{W}$。

## 一、前向傳播 (Forward Propagation)

MLP 包含輸入層、隱藏層和輸出層。第 $l$ 層的計算過程:

**1. 線性組合:**

$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$

**2. 激活函數:**

$$\mathbf{a}^{(l)} = f(\mathbf{z}^{(l)})$$

其中 $\mathbf{W}^{(l)}$ 是連接第 $l-1$ 層到第 $l$ 層的權重矩陣,$\mathbf{a}^{(0)} = \mathbf{x}$ 為輸入。

---

## 二、損失函數與梯度下降

### 2.1 損失函數 (Loss Function)

對於單個樣本,常用均方誤差 (MSE):

$$J = \frac{1}{2} \|\hat{\mathbf{y}} - \mathbf{y}\|^2 = \frac{1}{2} \sum_k (\hat{y}_k - y_k)^2$$

整個訓練集的損失為 $L = \sum_{n=1}^{N} J^{(n)}$。

### 2.2 梯度下降與權重調整公式

MLP 通過**梯度下降 (Gradient Descent)** 迭代更新權重:

$$\mathbf{W}^{(l)}_{\text{new}} = \mathbf{W}^{(l)}_{\text{old}} - \eta \frac{\partial L}{\partial \mathbf{W}^{(l)}}$$

$$\mathbf{b}^{(l)}_{\text{new}} = \mathbf{b}^{(l)}_{\text{old}} - \eta \frac{\partial L}{\partial \mathbf{b}^{(l)}}$$

其中:
- $\eta$ 是**學習率 (Learning Rate)**
- $\Delta \mathbf{W}^{(l)} = -\eta \frac{\partial L}{\partial \mathbf{W}^{(l)}}$ 是權重調整量

**關鍵在於高效計算梯度** $\frac{\partial L}{\partial \mathbf{W}^{(l)}}$。

---

## 三、反向傳播 (Backpropagation) 推導

反向傳播利用**鏈式法則**,從輸出層開始逐層向前計算梯度。

### 3.1 定義誤差項 $\boldsymbol{\delta}^{(l)}$

定義第 $l$ 層的誤差項為損失對線性組合的偏微分:

$$\boldsymbol{\delta}^{(l)} = \frac{\partial L}{\partial \mathbf{z}^{(l)}}$$

### 3.2 輸出層 (Layer $L$) 的誤差計算

$$\boldsymbol{\delta}^{(L)} = \frac{\partial L}{\partial \mathbf{z}^{(L)}} = \frac{\partial L}{\partial \mathbf{a}^{(L)}} \odot f'(\mathbf{z}^{(L)})$$

對於 MSE 損失和恆等激活函數:

$$\boldsymbol{\delta}^{(L)} = (\hat{\mathbf{y}} - \mathbf{y}) \odot f'(\mathbf{z}^{(L)})$$

### 3.3 隱藏層誤差的反向傳播

對於隱藏層 $l < L$,誤差從下一層傳播回來:

$$\boldsymbol{\delta}^{(l)} = \left[(\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)}\right] \odot f'(\mathbf{z}^{(l)})$$

其中 $\odot$ 表示逐元素乘法 (Hadamard product)。

### 3.4 權重和偏置的梯度

**1. 權重梯度:**

$$\frac{\partial L}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T$$

**2. 偏置梯度:**

$$\frac{\partial L}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$$

---

## 四、權重調整的實現流程

完整的訓練迭代過程 (實現 $\mathbf{W}^* = \mathbf{W} + \Delta \mathbf{W}$):

### 步驟 1: 初始化
隨機初始化所有 $\mathbf{W}^{(l)}$ 和 $\mathbf{b}^{(l)}$。

### 步驟 2: 前向傳播
輸入 $\mathbf{x}$,逐層計算:
- $\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$
- $\mathbf{a}^{(l)} = f(\mathbf{z}^{(l)})$

### 步驟 3: 計算損失
根據輸出 $\hat{\mathbf{y}} = \mathbf{a}^{(L)}$ 和真實值 $\mathbf{y}$ 計算損失 $L$。

### 步驟 4: 反向傳播
- 計算輸出層誤差: $\boldsymbol{\delta}^{(L)} = \frac{\partial L}{\partial \mathbf{a}^{(L)}} \odot f'(\mathbf{z}^{(L)})$
- 逐層向前計算: $\boldsymbol{\delta}^{(l)} = [(\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)}] \odot f'(\mathbf{z}^{(l)})$

### 步驟 5: 計算梯度
對每一層 $l$:
- $\frac{\partial L}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T$
- $\frac{\partial L}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$

### 步驟 6: 更新權重 (實現 $\mathbf{W}^*$)

$$\mathbf{W}^{(l)}_{\text{new}} = \mathbf{W}^{(l)}_{\text{old}} - \eta \frac{\partial L}{\partial \mathbf{W}^{(l)}}$$

$$\mathbf{b}^{(l)}_{\text{new}} = \mathbf{b}^{(l)}_{\text{old}} - \eta \frac{\partial L}{\partial \mathbf{b}^{(l)}}$$

這等價於:

$$\mathbf{W}^{(l)}_{\text{new}} = \mathbf{W}^{(l)}_{\text{old}} + \Delta \mathbf{W}^{(l)}$$

其中 $\Delta \mathbf{W}^{(l)} = -\eta \frac{\partial L}{\partial \mathbf{W}^{(l)}}$。

### 步驟 7: 迭代
重複步驟 2-6,直到損失收斂或達到預設訓練輪數。

---

## 五、SVM 與 MLP 權重調整的對比

| 特性 | SVM | MLP |
|------|-----|-----|
| **優化方法** | 二次規劃 (QP) | 梯度下降 |
| **權重更新** | 一次性求解 $\mathbf{w}^* = \sum \alpha_i^* y_i \mathbf{x}_i$ | 迭代更新 $\mathbf{W} = \mathbf{W} - \eta \nabla L$ |
| **關鍵算法** | SMO 算法 | 反向傳播 |
| **調整方式** | 直接計算最優解 | 逐步逼近最優解 |
| **依賴數據** | 僅支持向量 | 所有訓練樣本 |

# Colab上實作


