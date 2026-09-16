# Custom ResNet — Learning From Scratch

A learning-focused implementation of **CNNs and ResNets using PyTorch**, built to understand the mathematics and internal architecture instead of treating the model as a black box.

The project explores **convolution filters, strides, feature maps, matrix operations, backpropagation, gradient descent, and residual connections**.

## Objective

The model is designed around a 5-class diabetic retinopathy classification problem:

```text
0 → No DR

1 → Mild

2 → Moderate

3 → Severe

4 → Proliferative DR
```

The final model produces a probability distribution across the five classes.

```text
[0.03, 0.07, 0.72, 0.14, 0.04]
```

---

## Mathematical Foundation

### 1. Image

A `64 × 64` RGB image can be represented as:

$$
X \in \mathbb{R}^{3\times64\times64}
$$

giving:

$$
3\times64\times64=12288
$$

pixel values.

---

### 2. Convolution

A convolution filter is a set of trainable weights.

For an RGB `3×3` filter:

$$
W\in\mathbb{R}^{3\times3\times3}
$$

The filter slides over the image and performs multiplication and summation.

The output dimensions of a convolution are calculated as:

$$
H_{new} =
\left\lfloor
\frac{H_{old}+2P-K}{S}
\right\rfloor+1
$$

$$
W_{new} =
\left\lfloor
\frac{W_{old}+2P-K}{S}
\right\rfloor+1
$$

where:

```text
H_old / W_old → Initial height / width
K             → Kernel size
P             → Padding
S             → Stride
```

For example, with a `3×3` kernel, `0` padding, and stride `1`:

$$
\frac{64-3}{1}+1=62
$$

so:

```text
64 × 64 → 62 × 62
```

The filter performs the following operation:

$$
F(i,j)=
\sum_c\sum_u\sum_v
X(c,i+u,j+v)W(c,u,v)
$$

The result is a **feature map**.

Multiple filters produce multiple feature maps.

---

### 3. Activation

ReLU introduces non-linearity:

$$
ReLU(x)=\max(0,x)
$$

For example:

```text
[-2, 4, -1, 7]
```

becomes:

```text
[0, 4, 0, 7]
```

---

### 4. Classification

After feature extraction, the network performs:

$$
Z=XW+b
$$

producing five logits.

Softmax converts them into probabilities:

$$
P_i=
\frac{e^{Z_i}}{\sum_j e^{Z_j}}
$$

Example:

```text
No DR          0.03

Mild           0.07

Moderate       0.72

Severe         0.14

Proliferative  0.04
```

---

### 5. Learning

The dataset provides the correct class:

```text
Moderate DR → y = 2
```

The model prediction is compared against the target using a loss function.

The gradient of the loss with respect to each parameter is calculated:

$$
\frac{\partial L}{\partial W}
$$

The parameters are then updated using gradient descent:

$$
W_{new}
=
W_{old}
-
\eta
\frac{\partial L}{\partial W}
$$

This process updates the **convolution filters, weights, and biases**.

---

## Residual Learning

The main idea behind ResNet is the **identity/skip connection**.

Instead of only learning:

$$
H(x)
$$

a residual block learns:

$$
F(x)=H(x)-x
$$

so:

$$
\boxed{H(x)=F(x)+x}
$$

Conceptually:

```text
                 ┌──── Identity ────┐
                 │                   │
Input x ──► Conv ──► ReLU ──► Conv ──► (+)
                 │                   ▲
                 └───────────────────┘
```

After the two convolutional layers of a basic residual block, the transformed features are added to the original input through the identity path.

This allows the block to learn the **residual relationship** between the input and the transformed features.

If dimensions change, a `1×1` convolution can transform the identity branch so that the tensors can be added.

---

## Learning Roadmap

```text
Matrix Operations
       ↓
Trainable Weights
       ↓
Manual Convolution
       ↓
Strides & Feature Maps
       ↓
ReLU
       ↓
Classification
       ↓
Softmax
       ↓
Backpropagation
       ↓
Identity Connections
       ↓
Residual Blocks
       ↓
Custom ResNet
```

**The main purpose of this project is learning — understanding the mathematics behind CNNs and ResNets by implementing the concepts step by step.**
