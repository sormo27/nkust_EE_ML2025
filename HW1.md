# 支持向量機 (Support Vector Machine, SVM) 訓練過程與最佳權重 $W$ 的推導

本文詳細推導了線性可分支持向量機的訓練過程，核心目標是找到最大化分類間隔的超平面參數 $\mathbf{W}$ 和 $b$。

## 一、 SVM 最優化問題的定義

SVM 的訓練目標是尋找一個超平面 $\mathbf{W} \cdot \mathbf{x} + b = 0$，使得訓練集中的所有點都能被正確分類，且**間隔 (Margin)** 達到最大。

### 1.1 約束條件

對於所有訓練樣本 $(\mathbf{x}_i, y_i)$，必須滿足函數間隔大於等於 1 的約束：

$$y_i (\mathbf{W} \cdot \mathbf{x}_i + b) \ge 1, \quad i = 1, \dots, N$$

其中 $y_i \in \{-1, +1\}$ 是類別標籤。

### 1.2 間隔最大化

幾何間隔 (Geometric Margin) 為 $\gamma = \frac{1}{\|\mathbf{W}\|}$。最大化 $\gamma$ 等價於最小化 $\|\mathbf{W}\|$。

**原始最優化問題 (Primal Problem):**

$$\min_{\mathbf{W}, b} \frac{1}{2} \|\mathbf{W}\|^2 \quad \text{s.t.} \quad y_i (\mathbf{W} \cdot \mathbf{x}_i + b) - 1 \ge 0$$

## 二、 拉格朗日對偶問題推導

為了解決帶約束的優化問題，我們引入拉格朗日乘子 $\alpha_i \ge 0$。

### 2.1 構造拉格朗日函數

$$L(\mathbf{W}, b, \mathbf{\alpha}) = \frac{1}{2} \|\mathbf{W}\|^2 - \sum_{i=1}^{N} \alpha_i [y_i (\mathbf{W} \cdot \mathbf{x}_i + b) - 1]$$

### 2.2 求解 $\mathbf{W}^*$ 和 $b^*$ 的條件（KKT 條件）

我們對 $L$ 關於 $\mathbf{W}$ 和 $b$ 求偏導數，並令其為零：

1.  **對 $\mathbf{W}$ 求偏導：**
    $$\frac{\partial L}{\partial \mathbf{W}} = \mathbf{W} - \sum_{i=1}^{N} \alpha_i y_i \mathbf{x}_i = 0$$
    $$\implies \mathbf{W}^* = \sum_{i=1}^{N} \alpha_i y_i \mathbf{x}_i$$

2.  **對 $b$ 求偏導：**
    $$\frac{\partial L}{\partial b} = - \sum_{i=1}^{N} \alpha_i y_i = 0$$
    $$\implies \sum_{i=1}^{N} \alpha_i y_i = 0$$

### 2.3 對偶問題 (Dual Problem)

將 $\mathbf{W}^*$ 和 $\frac{\partial L}{\partial b}=0$ 代回 $L(\mathbf{W}, b, \mathbf{\alpha})$ 中，即可得到只關於 $\mathbf{\alpha}$ 的對偶問題：

$$\max_{\mathbf{\alpha}} \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (\mathbf{x}_i \cdot \mathbf{x}_j)$$
$$\text{s.t.} \quad \sum_{i=1}^{N} \alpha_i y_i = 0, \quad \alpha_i \ge 0, \quad i = 1, \dots, N$$

## 三、 最佳 $\mathbf{W}$ 的實現與調整 $(\mathbf{W}^*)$

對偶問題是一個**二次規劃 (QP)** 問題，通常使用 **SMO (Sequential Minimal Optimization)** 算法來高效求解最佳的 $\mathbf{\alpha}^*$ 向量。

### 3.1 求解 $\mathbf{\alpha}^*$

SMO 算法透過每次選擇兩個 $\alpha_i, \alpha_j$ 進行優化，將大規模的 QP 問題分解為一系列解析可解的 $2 \times 2$ 子問題。訓練過程就是不斷迭代，直到 $\mathbf{\alpha}$ 滿足 KKT 條件為止。

### 3.2 最佳權重向量 $\mathbf{W}^*$ 的確定 $(\mathbf{W}^* = \sum \alpha_i y_i \mathbf{x}_i)$

當 $\mathbf{\alpha}^*$ 被求出後，最佳權重向量 $\mathbf{W}^*$ 即可透過以下公式實現：

$$\mathbf{W}^* = \sum_{i=1}^{N} \alpha_i^* y_i \mathbf{x}_i$$

**權重調整的完成：**
在 SVM 中，訓練過程的目標**不是**透過微小的 $\Delta \mathbf{W}$ 逐步迭代調整 $\mathbf{W}$ (如梯度下降)，而是**直接**通過求解 $\mathbf{\alpha}^*$ **一次性**確定 $\mathbf{W}^*$。只有 $\alpha_i^* > 0$ 的樣本（即**支持向量**）才會對 $\mathbf{W}^*$ 的計算有貢獻。

### 3.3 最佳偏置 $b^*$ 的確定

選取任意一個滿足 $0 < \alpha_k^* < C$ （對於軟間隔 SVM）或 $\alpha_k^* > 0$（對於硬間隔 SVM）的支持向量 $\mathbf{x}_k$，即可解出 $b^*$：

$$b^* = y_k - \mathbf{W}^* \cdot \mathbf{x}_k$$

# 多層感知器 (MLP) 訓練過程：反向傳播 (Backpropagation) 推導與權重調整

本文詳細推導了使用反向傳播算法 (Backpropagation) 訓練 MLP 的過程，重點闡述了如何計算梯度以及如何實現權重的迭代調整 $\mathbf{W}^* = \mathbf{W} + \Delta \mathbf{W}$。

## 一、 前向傳播 (Forward Propagation) 概述

MLP 包含輸入層、隱藏層和輸出層。第 $l$ 層的計算過程如下：

1.  **線性組合：**
    $$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$
2.  **激活函數：**
    $$\mathbf{a}^{(l)} = f(\mathbf{z}^{(l)})$$
    其中 $\mathbf{W}^{(l)}$ 是連接 $l-1$ 層到 $l$ 層的權重矩陣。

## 二、 損失函數與梯度下降

### 2.1 損失函數 (Loss Function)

我們使用 $L$ 來表示整個訓練集上的總損失，目標是最小化 $L$。例如，對於一個單個樣本的均方誤差 (MSE)：

$$J = \frac{1}{2} (\hat{y} - y)^2$$

### 2.2 梯度下降與權重調整公式

MLP 的訓練是通過**梯度下降 (Gradient Descent)** 迭代更新權重 $\mathbf{W}$ 和偏置 $\mathbf{b}$ 來實現的。

**權重調整實現： $\mathbf{W}^* = \mathbf{W} + \Delta \mathbf{W}$**

$$\mathbf{W}_{\text{new}}^{(l)} = \mathbf{W}_{\text{old}}^{(l)} - \eta \frac{\partial L}{\partial \mathbf{W}^{(l)}}$$

$$\mathbf{b}_{\text{new}}^{(l)} = \mathbf{b}_{\text{old}}^{(l)} - \eta \frac{\partial L}{\partial \mathbf{b}^{(l)}}$$

其中 $\eta$ 是**學習率 (Learning Rate)**。**關鍵在於計算梯度** $\frac{\partial L}{\partial \mathbf{W}^{(l)}}$。

## 三、 反向傳播 (Backpropagation) 推導

反向傳播算法利用**鏈式法則**，從輸出層開始，逐層向前高效計算損失函數 $L$ 對每層權重 $\mathbf{W}^{(l)}$ 的偏導數。

### 3.1 輸出層 (Layer $L$) 的梯度計算

首先定義**誤差項 $\mathbf{\delta}^{(L)}$**，即損失對線性組合 $\mathbf{z}^{(L)}$ 的偏導：

$$\mathbf{\delta}^{(L)} = \frac{\partial L}{\partial \mathbf{z}^{(L)}} = \frac{\partial L}{\partial \mathbf{a}^{(L)}} \odot f'(\mathbf{z}^{(L)})$$

### 3.2 隱藏層 (Layer $l$) 的誤差傳播

對於隱藏層 $l < L$，誤差 $\mathbf{\delta}^{(l)}$ 是從下一層 $(l+1)$ 反向傳播回來的：

$$\mathbf{\delta}^{(l)} = ((\mathbf{W}^{(l+1)})^T \mathbf{\delta}^{(l+1)}) \odot f'(\mathbf{z}^{(l)})$$

### 3.3 權重和偏置的梯度計算

一旦計算出所有層的 $\mathbf{\delta}^{(l)}$，即可計算權重和偏置的梯度：

1.  **權重 $\mathbf{W}^{(l)}$ 的梯度：**
    $$\frac{\partial L}{\partial \mathbf{W}^{(l)}} = \mathbf{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T$$

2.  **偏置 $\mathbf{b}^{(l)}$ 的梯度：**
    $$\frac{\partial L}{\partial \mathbf{b}^{(l)}} = \mathbf{\delta}^{(l)}$$

## 四、 訓練過程的實現細節 (權重 $\mathbf{W}$ 的調整)

**調整的實現** $\mathbf{W}^* = \mathbf{W} + \Delta \mathbf{W}$ **流程：**

1.  **初始化：** 隨機初始化 $\mathbf{W}$ 和 $\mathbf{b}$。
2.  **前向傳播：** 輸入 $\mathbf{x}$，計算所有 $\mathbf{z}^{(l)}$ 和 $\mathbf{a}^{(l)}$。
3.  **計算損失：** 根據 $\hat{y} = \mathbf{a}^{(L)}$ 和真實值 $y$ 計算 $L$。
4.  **反向傳播：**
    * 計算輸出層的 $\mathbf{\delta}^{(L)}$。
    * 利用公式 $3.2$ 逐層向前計算所有隱藏層的 $\mathbf{\delta}^{(l)}$。
5.  **計算 $\Delta \mathbf{W}$：**
    * 對於每一層 $l$，計算梯度 $\mathbf{G}^{(l)} = \frac{\partial L}{\partial \mathbf{W}^{(l)}}$。
    * 計算調整量 $\mathbf{\Delta W}^{(l)} = - \eta \mathbf{G}^{(l)}$。
6.  **更新權重 (實現 $\mathbf{W}^*$)：**
    $$\mathbf{W}_{\text{new}}^{(l)} \leftarrow \mathbf{W}_{\text{old}}^{(l)} + \mathbf{\Delta W}^{(l)}$$
    $$\mathbf{b}_{\text{new}}^{(l)} \leftarrow \mathbf{b}_{\text{old}}^{(l)} - \eta \mathbf{\delta}^{(l)}$$
7.  **迭代：** 重複步驟 2-6，直到損失收斂或達到預設的訓練次數。
