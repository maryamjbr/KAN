# Kolmogorov--Arnold Network with Learnable B-Splines

An educational PyTorch implementation of a Kolmogorov--Arnold Network (KAN),
inspired by the original [KAN paper](https://arxiv.org/abs/2404.19756). The
project demonstrates the central architectural idea: instead of applying a
fixed activation at each node, every connection learns its own univariate
function.

The implementation is trained and evaluated on the Iris classification
dataset in `KAN.ipynb`.

## What is implemented

- A learnable activation on every input-output edge.
- Cubic B-spline bases evaluated with iterative Cox--de Boor dynamic
  programming.
- A trainable coefficient for **every** B-spline basis function.
- A learnable SiLU residual term for each edge.
- Uniform extended knot grids stored as PyTorch buffers, so they move with the
  model between CPU and GPU and are included in its state dictionary.
- Stacked KAN layers for multiclass classification.
- Automated checks for basis construction, coefficient gradients, tensor
  shapes, registered buffers, and end-to-end backpropagation.

## Edge-function formulation

Each edge learns

```text
phi(x) = w_base * SiLU(x) + sum_i c_i B_i,p(x),
```

where `c_i` are trainable coefficients and `B_i,p` is the B-spline basis of
degree `p`. The bases are built bottom-up from

```text
B_i,0(x) = 1  if t_i <= x < t_(i+1), otherwise 0,
```

using the Cox--de Boor recurrence

```text
B_i,p(x) = (x - t_i) / (t_(i+p) - t_i) * B_i,p-1(x)
         + (t_(i+p+1) - x) / (t_(i+p+1) - t_(i+1)) * B_i+1,p-1(x).
```

The dynamic-programming implementation computes all bases at one degree
before proceeding to the next. The final spline value is the weighted sum of
the complete degree-`p` basis, not a single basis function.

## Iris experiment

The notebook uses a reproducible, stratified 70/30 train-test split:

1. Standardize the four Iris features using statistics fitted on the training
   set only.
2. Train a `4 -> 8 -> 3` KAN with cubic splines and a 16-interval grid.
3. Evaluate train and test cross-entropy and accuracy.
4. Run the correctness test suite.
5. Plot one learned edge function together with its B-spline and SiLU
   components.

Iris is intentionally small: the goal is to make the spline mechanics and
gradient flow easy to inspect, not to claim a state-of-the-art benchmark.

### Verified result

With the configuration and seed stored in the notebook, the latest verified
CPU run reached **100.00% training accuracy** and **95.56% test accuracy
(43/45 samples)** after 20 epochs. The test set is small, so this number should
be treated as a reproducibility check rather than a broad performance claim.

## Installation

```bash
git clone https://github.com/maryamjbr/KAN.git
cd KAN
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the project

Start Jupyter and execute `KAN.ipynb` from top to bottom:

```bash
jupyter notebook KAN.ipynb
```

Run the tests independently with:

```bash
python -m unittest -v
```

## Repository structure

```text
KAN.ipynb        Reproducible Iris experiment, tests, and visualization
kan.py           B-spline activation, KAN layer, and stacked KAN model
test_kan.py      Automated correctness and gradient-flow tests
requirements.txt Runtime dependencies
README.md        Project documentation
```

## Scope and limitations

This is a compact, from-scratch implementation for learning and experimentation,
not a drop-in reproduction of the full reference KAN library. It does not
implement adaptive grid updates, pruning, symbolic regression, or the
regularization and interpretability tooling available in the authors' package.

## Reference

Z. Liu et al., *KAN: Kolmogorov--Arnold Networks*, 2024.
[arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
