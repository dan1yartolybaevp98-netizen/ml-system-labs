# Lab 5 — Frameworks and Automatic Differentiation

## 1. Introduction

This laboratory focused on building a scalar automatic differentiation system, checking its gradients against PyTorch and numerical calculations, training a small neural network, and examining PyTorch execution modes and tensor memory layouts.

All experiments were carried out in Python using a self-developed automatic differentiation engine together with PyTorch.

## 2. Developing a Custom Automatic Differentiation Engine

I created a custom `Value` class in `engine.py` to represent values in the computation graph.

Each node contains the following information:

- `data` — the scalar value stored in the node.
- `grad` — the gradient accumulated for that value.
- `_prev` — references to the parent nodes.
- `_op` — the operation responsible for creating the node.
- `_backward` — the function used to pass gradients to the previous nodes.

The engine supports these mathematical operations:

- Addition (`+`)
- Multiplication (`*`)
- Exponentiation (`**`)
- ReLU activation

The computation graph is generated automatically while the forward calculations are performed.

## 3. Backward Pass

To perform backpropagation, the engine first uses topological sorting to establish the correct order for processing the computation graph.

The backward procedure follows these steps:

1. Creates a topological ordering of all nodes in the graph.
2. Sets the output node's gradient to 1.
3. Processes the nodes in reverse topological order.
4. Propagates and combines gradients according to the chain rule.

Gradients are accumulated by addition instead of being overwritten. This makes it possible to correctly combine contributions from multiple paths in the computation graph.

The following function was used for testing:

\[
f=(ab+c)^2
\]

With \(a=2\), \(b=3\), and \(c=1\), the function returned 49.

The resulting derivatives were:

\[
\frac{\partial f}{\partial a}=42
\]

\[
\frac{\partial f}{\partial b}=28
\]

\[
\frac{\partial f}{\partial c}=14
\]

The calculated values agreed with the analytical derivatives.

## 4. Gradient Validation

I checked the custom engine by comparing its gradients with those produced by PyTorch Autograd and numerical finite-difference calculations.

The function used for this validation was:

\[
f=((ab+c)^2).ReLU()
\]

The input values were \(a=2\), \(b=3\), and \(c=1\).

### Gradient comparison

| Variable | Custom engine | PyTorch | Numerical |
|---|---:|---:|---:|
| a | 42.00000000 | 42.00000000 | 42.00000000 |
| b | 28.00000000 | 28.00000000 | 28.00000000 |
| c | 14.00000000 | 14.00000000 | 14.00000000 |

The largest absolute differences were:

- Custom engine compared with PyTorch: 0.0
- Custom engine compared with numerical gradients: 3.9137830754043534e-09

Both differences were smaller than the required tolerance of \(10^{-6}\).

**Result:** Every gradient validation test passed.

## 5. Training a Neural Network with make_moons

A small neural network was trained on the `make_moons` dataset.

### Dataset and network configuration

- Dataset: `make_moons`
- Total samples: 200
- Noise level: 0.2
- Random state: 42
- Network architecture: 2 → 8 → 1
- Hidden-layer activation: ReLU
- Output activation: Sigmoid
- Loss function: Binary Cross-Entropy
- Number of epochs: 40
- Learning rate: 0.05
- Benchmark repetitions: 5

The training performance of the custom engine and the PyTorch implementation was measured separately.

### Benchmark results

| Metric | Custom engine | PyTorch |
|---|---:|---:|
| Median training time | 4.82358 s | 0.02767 s |
| Final loss | 0.65017 | 0.56558 |

The speed comparison was calculated as follows:

\[
\frac{4.82358}{0.02767}\approx174.31
\]

Based on these measurements, the custom implementation took approximately 174 times longer than PyTorch.

This is expected because the custom engine handles individual scalar operations through Python objects, whereas PyTorch relies on optimized tensor operations and lower-level implementations.

PyTorch also produced a smaller final loss in this experiment.

## 6. PyTorch Computation Graph and Execution Modes

For this part, I built a small PyTorch model with the following structure:

\[
10 \rightarrow 16 \rightarrow 1
\]

The input tensor had a shape of `[32, 10]`, while the output tensor had a shape of `[32, 1]`.

I examined the computation graph using `torch.jit.trace`.

### Execution time comparison

| Execution mode | Median time |
|---|---:|
| Eager execution | 0.00002885 s |
| JIT Trace | 0.00002220 s |
| torch.compile | Not available |

In this test, JIT Trace ran approximately 23% faster than eager execution.

I could not complete the `torch.compile` test because the environment did not have a functioning C++ compiler. PyTorch returned an `InvalidCxxCompiler` error because `g++` was not available.

As a result, JIT tracing was used as the alternative execution mode.

## 7. Contiguous and Non-contiguous Tensors

In this experiment, I compared two tensors that had identical shapes and data types.

The non-contiguous version was produced by applying a transpose operation.

### Tensor properties

| Property | Contiguous | Non-contiguous |
|---|---|---|
| Shape | [1024, 1024] | [1024, 1024] |
| Data type | float32 | float32 |
| Stride | (1024, 1) | (1, 1024) |
| Contiguous | True | False |

Both tensors were tested with the same operation:

```python
(tensor * 1.5 + 2.0).relu()
```

### Performance measurements

| Tensor type | Median time |
|---|---:|
| Contiguous | 0.00240515 s |
| Non-contiguous | 0.00170695 s |

The measured time ratio was:

\[
\frac{T_{non-contiguous}}{T_{contiguous}}=0.71
\]

For this particular test, the non-contiguous tensor completed the operation faster.

However, this observation only applies to the tested operation, tensor dimensions, and hardware. Non-contiguous tensors are not always faster, since performance is influenced by memory access patterns and the type of operation.

## 8. Conclusion

This laboratory provided practical experience with automatic differentiation and the way PyTorch executes neural network operations.

The custom engine was able to build computation graphs and calculate gradients using reverse-mode automatic differentiation. Its results matched PyTorch Autograd exactly and were also within the required tolerance when compared with numerical estimates.

The neural network benchmark highlighted the performance gap between a scalar implementation written in Python and an optimized tensor framework. In the measured experiment, PyTorch was approximately 174 times faster during training.

The execution-mode comparison showed a small speed improvement when using JIT Trace instead of eager execution. The `torch.compile` test was unavailable because the environment did not contain a working C++ compiler.

Finally, the tensor-layout experiment showed that memory arrangement and stride values can influence execution time. The observed performance difference depended on the particular operation and hardware used.

Overall, the work covered computation graphs, automatic differentiation, gradient verification, neural network training, PyTorch execution modes, and tensor memory layouts.
