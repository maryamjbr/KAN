# Kolmogorov--Arnold Network with Learnable B-Splines

An educational PyTorch implementation of a Kolmogorov--Arnold Network (KAN),
inspired by the original [KAN paper](https://arxiv.org/abs/2404.19756). The
project demonstrates the central architectural idea: instead of applying a
fixed activation at each node, every connection learns its own univariate
function.

The implementation is trained and evaluated against multilayer perceptron
(MLP) baselines on Iris in `KAN.ipynb` and on the larger Digits dataset in
`benchmark_digits.py`.

## What is implemented

- A learnable activation on every input-output edge.
- Cubic B-spline bases evaluated with iterative Cox--de Boor dynamic
  programming.
- A trainable coefficient for **every** B-spline basis function.
- A learnable SiLU residual term for each edge.
- Uniform extended knot grids stored as PyTorch buffers, so they move with the
  model between CPU and GPU and are included in its state dictionary.
- Stacked KAN layers for multiclass classification.
- A five-seed Digits benchmark with width-matched and parameter-matched MLP
  baselines.
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

The notebook uses reproducible, stratified data splits:

1. Reserve 30% of the data as a held-out test set, then use 20% of the
   remaining development data for validation (84/21/45 train/validation/test
   samples).
2. Standardize the four Iris features using statistics fitted on the training
   set only.
3. Train a `4 -> 8 -> 3` KAN with cubic splines and a 16-interval grid and a
   width-matched `4 -> 8 -> 3` ReLU MLP baseline.
4. Report validation loss and accuracy after epochs 1, 10, and 20, then
   evaluate the held-out test set once at the end.
5. Display a compact model/parameter/accuracy comparison table.
6. Run the correctness test suite.
7. Plot one learned edge function together with its B-spline and SiLU
   components.

Iris is intentionally small: the goal is to make the spline mechanics and
gradient flow easy to inspect, not to claim a state-of-the-art benchmark.

### Verified comparison

With the configuration and seed stored in the notebook, the latest verified
CPU run produced:

| Model | Parameters | Validation accuracy (epoch 20) | Final test accuracy |
|:--|--:|--:|--:|
| KAN | 1,131 | 95.24% (20/21) | 95.56% (43/45) |
| MLP | 67 | 90.48% (19/21) | 97.78% (44/45) |

These are single-split reproducibility results, not cross-validation estimates.
The validation and test sets are small, so a one-sample difference changes the
reported accuracy substantially and should not be treated as evidence that one
architecture is generally better.

## Digits benchmark

`benchmark_digits.py` provides a more meaningful comparison on scikit-learn's
1,797-sample, 64-feature, 10-class Digits dataset. For each of five seeds it:

1. Creates a fresh stratified split with 1,257 training, 270 validation, and
   270 held-out test samples.
2. Fits feature standardization on the training split only.
3. Trains each model for at most 50 epochs with the same Adam settings and
   shuffled batch order.
4. Compares the KAN with both a width-matched MLP and an approximately
   parameter-matched MLP.
5. Monitors validation loss every epoch, reports validation metrics at epochs
   1, 10, and 20, and stops after eight epochs without an improvement of at
   least `1e-4`.
6. Restores the checkpoint with the lowest validation loss and evaluates the
   test split exactly once.

### Verified five-seed result

The seeds are `1, 7, 21, 42, 2025`. Values are the mean and sample standard
deviation across the five paired repeated-holdout runs on CPU.

| Model | Parameters | Validation accuracy at best-loss epoch | Final test accuracy | Mean best epoch | Mean training time |
|:--|--:|--:|--:|--:|--:|
| KAN | 7,122 | 94.52% +/- 2.21% | 95.41% +/- 1.40% | 40.4 | 35.63s |
| MLP (width-matched) | 610 | 94.30% +/- 0.97% | 93.56% +/- 1.61% | 26.2 | 0.17s |
| MLP (parameter-matched) | 7,135 | 97.11% +/- 0.76% | 97.78% +/- 0.87% | 35.0 | 0.22s |

KAN and the much smaller width-matched MLP have similar accuracy under this
setup. At approximately the same parameter count, the MLP is more accurate and
far faster. KAN's validation and test means are close, which does not indicate
classic overfitting in this experiment. The timing difference mainly reflects
this educational KAN's edge-by-edge Python implementation; it is not a general
benchmark of all KAN implementations. Training times are machine-dependent.

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

Run the five-seed Digits benchmark with:

```bash
python benchmark_digits.py
```

For a quick smoke run or a custom experiment:

```bash
python benchmark_digits.py --epochs 20 --patience 8 --seeds 1
```

Run the tests independently with:

```bash
python -m unittest -v
```

## Repository structure

```text
KAN.ipynb          Reproducible Iris experiment, tests, and visualization
benchmark_digits.py Five-seed Digits benchmark against two MLP baselines
kan.py             B-spline activation, KAN layer, and stacked KAN model
test_benchmark_digits.py Benchmark split and parameter-budget tests
test_kan.py        Automated correctness and gradient-flow tests
requirements.txt   Runtime dependencies
README.md          Project documentation
```

## Scope and limitations

This is a compact, from-scratch implementation for learning and experimentation,
not a drop-in reproduction of the full reference KAN library. It does not
implement adaptive grid updates, pruning, symbolic regression, or the
regularization and interpretability tooling available in the authors' package.
`KANLayer` also evaluates edges individually, prioritizing clarity over the
vectorized performance needed for large-scale experiments.

## Reference

Z. Liu et al., *KAN: Kolmogorov--Arnold Networks*, 2024.
[arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
