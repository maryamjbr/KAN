"""Reproducible KAN/MLP benchmark on scikit-learn's Digits dataset.

The benchmark uses paired, stratified train/validation/test splits for every
seed.  It compares the KAN with both a width-matched MLP and an approximately
parameter-matched MLP.  The test split is evaluated only after the fixed
training budget has finished.
"""

from __future__ import annotations

import argparse
import copy
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from kan import KAN


INPUT_DIM = 64
OUTPUT_DIM = 10
HIDDEN_DIM = 8
GRID_SIZE = 8
DEGREE = 3
BATCH_SIZE = 128
DEFAULT_SEEDS = (1, 7, 21, 42, 2025)


@dataclass(frozen=True)
class RunResult:
    parameters: int
    validation_accuracy: float
    test_accuracy: float
    training_seconds: float
    best_epoch: int
    epochs_trained: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def as_dataset(features: np.ndarray, targets: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.long),
    )


def prepare_datasets(seed: int) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    features, targets = load_digits(return_X_y=True)
    development_features, test_features, development_targets, test_targets = (
        train_test_split(
            features,
            targets,
            test_size=0.15,
            random_state=seed,
            stratify=targets,
        )
    )
    train_features, validation_features, train_targets, validation_targets = (
        train_test_split(
            development_features,
            development_targets,
            test_size=270,
            random_state=seed,
            stratify=development_targets,
        )
    )

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    validation_features = scaler.transform(validation_features)
    test_features = scaler.transform(test_features)

    return (
        as_dataset(train_features, train_targets),
        as_dataset(validation_features, validation_targets),
        as_dataset(test_features, test_targets),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, targets in data_loader:
            logits = model(features)
            total_loss += criterion(logits, targets).item() * targets.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    return total_loss / total, correct / total


def train_once(
    name: str,
    model_factory: Callable[[], nn.Module],
    datasets: tuple[TensorDataset, TensorDataset, TensorDataset],
    seed: int,
    epochs: int,
    patience: int,
    min_delta: float,
) -> RunResult:
    set_seed(seed)
    train_dataset, validation_dataset, test_dataset = datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(validation_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    model = model_factory()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    reporting_epochs = {1, 10, 20}
    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    epochs_trained = 0
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epochs_trained = epoch
        model.train()
        for features, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()

        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, criterion
        )
        if validation_loss < best_validation_loss - min_delta:
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch in reporting_epochs:
            print(
                f"  {name:<24} epoch {epoch:02d} | "
                f"validation loss: {validation_loss:.4f} | "
                f"accuracy: {validation_accuracy:.2%}"
            )

        if epochs_without_improvement >= patience:
            print(
                f"  {name:<24} early stop at epoch {epoch:02d} | "
                f"best epoch: {best_epoch:02d}"
            )
            break

    training_seconds = time.perf_counter() - start_time
    if best_state is None:
        raise RuntimeError("training completed without a validation checkpoint")
    model.load_state_dict(best_state)
    _, test_accuracy = evaluate(model, test_loader, criterion)
    print(
        f"  {name:<24} restored epoch {best_epoch:02d} | "
        f"final test accuracy: {test_accuracy:.2%} | "
        f"time: {training_seconds:.2f}s"
    )
    return RunResult(
        parameters=count_parameters(model),
        validation_accuracy=best_validation_accuracy,
        test_accuracy=test_accuracy,
        training_seconds=training_seconds,
        best_epoch=best_epoch,
        epochs_trained=epochs_trained,
    )


def parameter_matched_hidden_dim(target_parameters: int) -> int:
    parameters_per_hidden_unit = INPUT_DIM + OUTPUT_DIM + 1
    return round((target_parameters - OUTPUT_DIM) / parameters_per_hidden_unit)


def model_factories() -> dict[str, Callable[[], nn.Module]]:
    kan_factory = lambda: KAN(  # noqa: E731 - concise experiment factory
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dims=[HIDDEN_DIM],
        grid_size=GRID_SIZE,
        degree=DEGREE,
        grid_range=(-3.0, 3.0),
    )
    target_parameters = count_parameters(kan_factory())
    matched_hidden_dim = parameter_matched_hidden_dim(target_parameters)

    return {
        "KAN": kan_factory,
        "MLP (width-matched)": lambda: nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        ),
        "MLP (parameter-matched)": lambda: nn.Sequential(
            nn.Linear(INPUT_DIM, matched_hidden_dim),
            nn.ReLU(),
            nn.Linear(matched_hidden_dim, OUTPUT_DIM),
        ),
    }


def mean_and_sample_std(values: Sequence[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def print_summary(results: dict[str, list[RunResult]]) -> None:
    print("\nRepeated-holdout summary (mean +/- sample standard deviation)")
    print(
        "| Model | Parameters | Validation accuracy at best-loss epoch | "
        "Final test accuracy | Mean best epoch | Mean training time |"
    )
    print("|:--|--:|--:|--:|--:|--:|")
    for name, model_results in results.items():
        validation_mean, validation_std = mean_and_sample_std(
            [result.validation_accuracy for result in model_results]
        )
        test_mean, test_std = mean_and_sample_std(
            [result.test_accuracy for result in model_results]
        )
        time_mean = statistics.mean(
            result.training_seconds for result in model_results
        )
        best_epoch_mean = statistics.mean(
            result.best_epoch for result in model_results
        )
        print(
            f"| {name} | {model_results[0].parameters:,} | "
            f"{validation_mean:.2%} +/- {validation_std:.2%} | "
            f"{test_mean:.2%} +/- {test_std:.2%} | {best_epoch_mean:.1f} | "
            f"{time_mean:.2f}s |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="maximum training epochs (default: 50)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="epochs without validation-loss improvement before stopping (default: 8)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="minimum validation-loss improvement (default: 1e-4)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.patience < 1:
        parser.error("--patience must be positive")
    if args.min_delta < 0:
        parser.error("--min-delta cannot be negative")
    if not args.seeds:
        parser.error("--seeds requires at least one integer")
    return args


def main() -> None:
    args = parse_args()
    factories = model_factories()
    results: dict[str, list[RunResult]] = {name: [] for name in factories}

    print(
        "Digits split sizes per seed: 1,257 train / 270 validation / 270 test"
    )
    print(
        f"Seeds: {args.seeds}; maximum epochs: {args.epochs}; "
        f"patience: {args.patience}; batch size: {BATCH_SIZE}"
    )
    for seed in args.seeds:
        print(f"\nSeed {seed}")
        datasets = prepare_datasets(seed)
        for name, factory in factories.items():
            results[name].append(
                train_once(
                    name,
                    factory,
                    datasets,
                    seed,
                    args.epochs,
                    args.patience,
                    args.min_delta,
                )
            )

    print_summary(results)


if __name__ == "__main__":
    main()
