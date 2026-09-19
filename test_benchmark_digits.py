import unittest

from benchmark_digits import count_parameters, model_factories, prepare_datasets


class TestDigitsBenchmark(unittest.TestCase):
    def test_split_sizes(self):
        train, validation, test = prepare_datasets(seed=1)

        self.assertEqual(len(train), 1257)
        self.assertEqual(len(validation), 270)
        self.assertEqual(len(test), 270)

    def test_model_parameter_budgets(self):
        factories = model_factories()
        parameter_counts = {
            name: count_parameters(factory()) for name, factory in factories.items()
        }

        self.assertEqual(parameter_counts["KAN"], 7122)
        self.assertEqual(parameter_counts["MLP (width-matched)"], 610)
        self.assertEqual(parameter_counts["MLP (parameter-matched)"], 7135)
        self.assertLess(
            abs(
                parameter_counts["KAN"]
                - parameter_counts["MLP (parameter-matched)"]
            ),
            parameter_counts["KAN"] * 0.01,
        )


if __name__ == "__main__":
    unittest.main()
