import unittest

import torch

from kan import BSplineActivation, KAN, KANLayer


class TestBSplineActivation(unittest.TestCase):
    def test_basis_is_a_partition_of_unity_on_grid(self):
        activation = BSplineActivation(grid_size=8, degree=3)
        inputs = torch.linspace(-3.0, 3.0, 101)

        basis = activation.b_spline_basis(inputs)

        self.assertEqual(basis.shape, (101, activation.n_basis))
        torch.testing.assert_close(
            basis.sum(dim=-1),
            torch.ones_like(inputs),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_all_spline_coefficients_receive_gradients(self):
        activation = BSplineActivation(grid_size=8, degree=3)
        inputs = torch.linspace(-3.0, 3.0, 257)

        activation.spline(inputs).sum().backward()

        self.assertIsNotNone(activation.coefficients.grad)
        self.assertTrue(torch.all(activation.coefficients.grad.abs() > 0))

    def test_knots_are_registered_as_a_buffer(self):
        activation = BSplineActivation()

        self.assertIn("knots", dict(activation.named_buffers()))
        self.assertNotIn("knots", dict(activation.named_parameters()))

    def test_activation_preserves_input_shape(self):
        activation = BSplineActivation()
        inputs = torch.randn(4, 3)

        self.assertEqual(activation(inputs).shape, inputs.shape)


class TestKANLayer(unittest.TestCase):
    def test_layer_output_shape(self):
        layer = KANLayer(in_features=3, out_features=2, grid_size=8)

        self.assertEqual(layer(torch.randn(5, 3)).shape, (5, 2))

    def test_layer_rejects_wrong_input_shape(self):
        layer = KANLayer(in_features=3, out_features=2)

        with self.assertRaisesRegex(ValueError, "expected shape"):
            layer(torch.randn(5, 4))


class TestKAN(unittest.TestCase):
    def test_end_to_end_backward_pass(self):
        model = KAN(
            input_dim=4,
            output_dim=3,
            hidden_dims=[5],
            grid_size=8,
        )
        inputs = torch.randn(7, 4)

        output = model(inputs)
        output.square().mean().backward()

        self.assertEqual(output.shape, (7, 3))
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()
