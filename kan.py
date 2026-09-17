"""Minimal educational Kolmogorov--Arnold Network components.

The implementation follows the KAN idea of placing a learnable univariate
function on every edge.  Each edge function combines a SiLU residual term
with a trainable B-spline expansion.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BSplineActivation(nn.Module):
    """Learnable edge function based on a uniform B-spline basis.

    ``grid_size`` is the number of intervals in ``grid_range``.  The knot
    vector is extended by ``degree`` intervals on both sides so that the
    basis functions form a partition of unity throughout the requested
    range.
    """

    def __init__(
        self,
        grid_size: int = 16,
        degree: int = 3,
        grid_range: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        super().__init__()
        if grid_size < 1:
            raise ValueError("grid_size must be at least 1")
        if degree < 0:
            raise ValueError("degree must be non-negative")
        if grid_range[0] >= grid_range[1]:
            raise ValueError("grid_range must be an increasing pair")

        self.grid_size = grid_size
        self.degree = degree
        self.grid_range = grid_range

        step = (grid_range[1] - grid_range[0]) / grid_size
        knot_indices = torch.arange(
            -degree,
            grid_size + degree + 1,
            dtype=torch.float32,
        )
        knots = grid_range[0] + step * knot_indices
        self.register_buffer("knots", knots)

        self.n_basis = knots.numel() - degree - 1
        self.coefficients = nn.Parameter(torch.empty(self.n_basis))
        self.base_weight = nn.Parameter(torch.ones(()))
        nn.init.normal_(self.coefficients, mean=0.0, std=0.02)

    def b_spline_basis(self, x: Tensor) -> Tensor:
        """Evaluate every B-spline basis function using Cox--de Boor DP.

        The returned tensor has shape ``(*x.shape, n_basis)``.  Building the
        table iteratively avoids the repeated work of a recursive evaluator
        while remaining differentiable with respect to ``x``.
        """

        if not torch.is_floating_point(x):
            raise TypeError("x must be a floating-point tensor")

        knots = self.knots.to(dtype=x.dtype)
        x_expanded = x.unsqueeze(-1)

        # Degree-zero basis functions: one for the interval containing x.
        basis = (
            (x_expanded >= knots[:-1]) & (x_expanded < knots[1:])
        ).to(dtype=x.dtype)

        # Each iteration constructs the next row of the Cox--de Boor table.
        for current_degree in range(1, self.degree + 1):
            n_current = knots.numel() - current_degree - 1

            left_denominator = (
                knots[current_degree : current_degree + n_current]
                - knots[:n_current]
            )
            right_denominator = (
                knots[current_degree + 1 : current_degree + 1 + n_current]
                - knots[1 : 1 + n_current]
            )

            left = (
                (x_expanded - knots[:n_current]) / left_denominator
            ) * basis[..., :n_current]
            right = (
                (
                    knots[
                        current_degree + 1 : current_degree + 1 + n_current
                    ]
                    - x_expanded
                )
                / right_denominator
            ) * basis[..., 1 : n_current + 1]
            basis = left + right

        return basis

    def spline(self, x: Tensor) -> Tensor:
        """Return the trainable weighted sum of all B-spline bases."""

        return torch.sum(
            self.b_spline_basis(x) * self.coefficients,
            dim=-1,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.base_weight * F.silu(x) + self.spline(x)


class KANLayer(nn.Module):
    """A layer whose input-output edges each own a spline activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 16,
        degree: int = 3,
        grid_range: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.activations = nn.ModuleList(
            BSplineActivation(grid_size, degree, grid_range)
            for _ in range(in_features * out_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"expected shape (batch, {self.in_features}), got {tuple(x.shape)}"
            )

        outputs = []
        for output_index in range(self.out_features):
            value = self.bias[output_index].expand(x.shape[0])
            for input_index in range(self.in_features):
                edge_index = output_index * self.in_features + input_index
                value = value + self.activations[edge_index](x[:, input_index])
            outputs.append(value)
        return torch.stack(outputs, dim=1)


class KAN(nn.Module):
    """Stacked KAN layers for small supervised-learning experiments."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        grid_size: int = 16,
        degree: int = 3,
        grid_range: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        super().__init__()
        dimensions = [input_dim, *hidden_dims, output_dim]
        self.model = nn.Sequential(
            *(
                KANLayer(
                    dimensions[index],
                    dimensions[index + 1],
                    grid_size,
                    degree,
                    grid_range,
                )
                for index in range(len(dimensions) - 1)
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
