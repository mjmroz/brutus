#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive tests for brutus mathematical utilities.

This test suite includes:
1. Unit tests for the new mathematical functions
2. Comparison tests between old and new implementations
3. Integration tests for matrix operations and statistical functions
"""

import numpy as np
import pytest
import scipy.stats

from brutus.utils.math import truncnorm_logpdf


class TestMatrixOperations:
    """Unit tests for 3x3 matrix operations."""

    def test_adjoint3_basic(self):
        """Test basic adjoint computation."""
        from brutus.utils.math import adjoint3

        # Simple test case: identity matrix
        I = np.eye(3)
        adj_I = adjoint3(I)

        # Adjoint of identity should be identity
        np.testing.assert_array_almost_equal(adj_I, I, decimal=12)

    def test_adjoint3_batch(self):
        """Test adjoint computation for batch of matrices."""
        from brutus.utils.math import adjoint3

        # Create batch of random 3x3 matrices
        np.random.seed(42)
        batch_size = 5
        A = np.random.random((batch_size, 3, 3))

        # Compute adjoint
        adj_A = adjoint3(A)

        # Check shape
        assert adj_A.shape == (batch_size, 3, 3)

        # For each matrix, A * adj(A) should equal det(A) * I
        for i in range(batch_size):
            det_A = np.linalg.det(A[i])
            product = np.dot(A[i], adj_A[i])
            expected = det_A * np.eye(3)
            np.testing.assert_array_almost_equal(product, expected, decimal=10)

    def test_dot3_basic(self):
        """Test basic dot product computation."""
        from brutus.utils.math import dot3

        # Simple test cases
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])

        result = dot3(a, b)
        expected = np.dot(a, b)  # Should be 32

        assert result == expected
        assert result == 32

    def test_dot3_batch(self):
        """Test dot product for batch of vectors."""
        from brutus.utils.math import dot3

        np.random.seed(42)
        batch_size = 10
        vector_size = 5

        A = np.random.random((batch_size, vector_size))
        B = np.random.random((batch_size, vector_size))

        result = dot3(A, B)
        expected = np.sum(A * B, axis=-1)

        np.testing.assert_array_almost_equal(result, expected, decimal=12)

    def test_inverse3_basic(self):
        """Test 3x3 matrix inversion."""
        from brutus.utils.math import inverse3

        # Create a simple invertible matrix
        A = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]], dtype=float)

        A_inv = inverse3(A)

        # Check that A * A_inv = I
        product = np.dot(A, A_inv)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(product, expected, decimal=12)

    def test_inverse3_vs_numpy(self):
        """Test inverse computation against numpy.linalg.inv."""
        from brutus.utils.math import inverse3

        np.random.seed(42)

        # Create random invertible matrices
        for _ in range(10):
            A = np.random.random((3, 3))
            A = A + A.T + 3 * np.eye(3)  # Make positive definite (invertible)

            custom_inv = inverse3(A)
            numpy_inv = np.linalg.inv(A)

            np.testing.assert_array_almost_equal(custom_inv, numpy_inv, decimal=10)

    def test_inverse3_batch(self):
        """Test batch matrix inversion."""
        from brutus.utils.math import inverse3

        np.random.seed(42)
        batch_size = 5

        # Create batch of random invertible matrices
        A = np.random.random((batch_size, 3, 3))
        A = A + np.transpose(A, (0, 2, 1)) + 3 * np.eye(3)[None, :, :]

        A_inv = inverse3(A)

        # Check that each A * A_inv = I
        for i in range(batch_size):
            product = np.dot(A[i], A_inv[i])
            expected = np.eye(3)
            np.testing.assert_array_almost_equal(product, expected, decimal=10)

    def test_isPSD_basic(self):
        """Test positive semidefinite check."""
        from brutus.utils.math import isPSD

        # Identity matrix is PSD
        I = np.eye(3)
        assert isPSD(I) == True

        # Zero matrix is PSD
        Z = np.zeros((3, 3))
        assert isPSD(Z) == True

        # Positive definite matrix is PSD
        A = np.array([[2, 1], [1, 2]], dtype=float)
        assert isPSD(A) == True

        # Non-PSD matrix
        B = np.array([[1, 2], [2, 1]], dtype=float)  # Has negative eigenvalue
        assert isPSD(B) == False


class TestStatisticalDistributions:
    """Unit tests for statistical distribution functions."""

    def test_chisquare_logpdf_basic(self):
        """Test chi-square log-PDF computation."""
        from brutus.utils.math import chisquare_logpdf

        # Test against scipy
        x = np.array([1.0, 2.0, 5.0, 10.0])
        df = 3

        custom_logpdf = chisquare_logpdf(x, df)
        scipy_logpdf = scipy.stats.chi2.logpdf(x, df)

        np.testing.assert_array_almost_equal(custom_logpdf, scipy_logpdf, decimal=10)

    def test_chisquare_logpdf_with_scaling(self):
        """Test chi-square log-PDF with location and scale."""
        from brutus.utils.math import chisquare_logpdf

        x = np.array([2.0, 4.0, 8.0])
        df = 2
        loc = 1.0
        scale = 2.0

        custom_logpdf = chisquare_logpdf(x, df, loc=loc, scale=scale)
        scipy_logpdf = scipy.stats.chi2.logpdf(x, df, loc=loc, scale=scale)

        np.testing.assert_array_almost_equal(custom_logpdf, scipy_logpdf, decimal=10)

    def test_chisquare_logpdf_edge_cases(self):
        """Test chi-square log-PDF edge cases."""
        from brutus.utils.math import chisquare_logpdf

        # Zero and negative values should give -inf
        x_neg = np.array([-1.0, 0.0])
        df = 2

        result = chisquare_logpdf(x_neg, df)
        assert np.all(result == -np.inf)

    def test_truncnorm_pdf_basic(self):
        """Test truncated normal PDF computation."""
        from brutus.utils.math import truncnorm_pdf

        x = np.array([-1.0, 0.0, 1.0, 2.0])
        a, b = -2.0, 2.0  # Truncation bounds

        custom_pdf = truncnorm_pdf(x, a, b)
        scipy_pdf = scipy.stats.truncnorm.pdf(x, a, b)

        np.testing.assert_array_almost_equal(custom_pdf, scipy_pdf, decimal=10)

    def test_truncnorm_pdf_with_scaling(self):
        """Test truncated normal PDF with location and scale."""
        from brutus.utils.math import truncnorm_pdf

        x = np.array([2.0, 3.0, 4.0, 5.0])
        a, b = -1.0, 1.0
        loc = 3.0
        scale = 0.5

        custom_pdf = truncnorm_pdf(x, a, b, loc=loc, scale=scale)
        scipy_pdf = scipy.stats.truncnorm.pdf(x, a, b, loc=loc, scale=scale)

        np.testing.assert_array_almost_equal(custom_pdf, scipy_pdf, decimal=10)

    def test_truncnorm_pdf_outside_bounds(self):
        """Test truncated normal PDF outside truncation bounds."""
        from brutus.utils.math import truncnorm_pdf

        # Values outside truncation bounds should have PDF = 0
        x = np.array([-5.0, 5.0])  # Outside [-2, 2]
        a, b = -2.0, 2.0

        result = truncnorm_pdf(x, a, b)
        expected = np.array([0.0, 0.0])

        np.testing.assert_array_equal(result, expected)

    def test_truncnorm_logpdf_basic(self):
        """Test truncated normal log-PDF computation."""
        from brutus.utils.math import truncnorm_logpdf

        x = np.array([-1.0, 0.0, 1.0])
        a, b = -2.0, 2.0

        custom_logpdf = truncnorm_logpdf(x, a, b)
        scipy_logpdf = scipy.stats.truncnorm.logpdf(x, a, b)

        np.testing.assert_array_almost_equal(custom_logpdf, scipy_logpdf, decimal=10)

    def test_truncnorm_logpdf_outside_bounds(self):
        """Test truncated normal log-PDF outside bounds."""
        from brutus.utils.math import truncnorm_logpdf

        # Values outside bounds should have log-PDF = -inf
        x = np.array([-5.0, 5.0])
        a, b = -2.0, 2.0

        result = truncnorm_logpdf(x, a, b)
        expected = np.array([-np.inf, -np.inf])

        np.testing.assert_array_equal(result, expected)


class TestFunctionWrapper:
    """Unit tests for the function wrapper utility."""

    def test_function_wrapper_basic(self):
        """Test basic function wrapper functionality."""
        from brutus.utils.math import _function_wrapper

        def test_func(x, a, b=2):
            return x * a + b

        wrapped = _function_wrapper(test_func, args=(3,), kwargs={"b": 5}, name="test")

        result = wrapped(10)
        expected = 10 * 3 + 5  # = 35

        assert result == expected

    def test_function_wrapper_error_handling(self):
        """Test function wrapper error handling."""
        from brutus.utils.math import _function_wrapper

        def failing_func(x):
            raise ValueError("Test error")

        wrapped = _function_wrapper(failing_func, args=(), kwargs={}, name="failing")

        with pytest.raises(ValueError, match="Test error"):
            wrapped(1)


class TestMathComparison:
    """Comparison tests between old and new implementations."""

    def test_adjoint3_vs_original(self):
        """Compare new _adjoint3 function with original."""
        try:
            from brutus.utilities import _adjoint3 as orig_adjoint3
        except ImportError:
            pytest.skip("Original utils.py not available for comparison")

        from brutus.utils.math import adjoint3 as new_adjoint3

        # Test data
        np.random.seed(42)
        A = np.random.random((3, 3, 3))

        # Compute with both implementations
        orig_result = orig_adjoint3(A)
        new_result = new_adjoint3(A)

        # Should be identical
        np.testing.assert_array_almost_equal(new_result, orig_result, decimal=12)

    def test_inverse3_vs_original(self):
        """Compare new _inverse3 function with original."""
        try:
            from brutus.utilities import _inverse3 as orig_inverse3
        except ImportError:
            pytest.skip("Original utils.py not available for comparison")

        from brutus.utils.math import inverse3 as new_inverse3

        # Test data - create invertible matrices
        np.random.seed(42)
        A = np.random.random((2, 3, 3))
        A = A + np.transpose(A, (0, 2, 1)) + 3 * np.eye(3)[None, :, :]

        # Compute with both implementations
        orig_result = orig_inverse3(A)
        new_result = new_inverse3(A)

        # Should be identical
        np.testing.assert_array_almost_equal(new_result, orig_result, decimal=12)

    def test_chisquare_logpdf_vs_original(self):
        """Compare new _chisquare_logpdf function with original."""
        try:
            from brutus.utilities import _chisquare_logpdf as orig_chisquare_logpdf
        except ImportError:
            pytest.skip("Original utils.py not available for comparison")

        from brutus.utils.math import chisquare_logpdf as new_chisquare_logpdf

        # Test data
        x = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        df = 3
        loc = 0.5
        scale = 1.5

        # Compute with both implementations
        orig_result = orig_chisquare_logpdf(x, df, loc=loc, scale=scale)
        new_result = new_chisquare_logpdf(x, df, loc=loc, scale=scale)

        # Should be identical
        np.testing.assert_array_almost_equal(new_result, orig_result, decimal=12)

    def test_truncnorm_pdf_vs_original(self):
        """Compare new _truncnorm_pdf function with original."""
        try:
            from brutus.utilities import _truncnorm_pdf as orig_truncnorm_pdf
        except ImportError:
            pytest.skip("Original utils.py not available for comparison")

        from brutus.utils.math import truncnorm_pdf as new_truncnorm_pdf

        # Test data
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        a, b = -2.0, 2.0
        loc = 0.5
        scale = 1.2

        # Compute with both implementations
        orig_result = orig_truncnorm_pdf(x, a, b, loc=loc, scale=scale)
        new_result = new_truncnorm_pdf(x, a, b, loc=loc, scale=scale)

        # Should be identical
        np.testing.assert_array_almost_equal(new_result, orig_result, decimal=12)


class TestMathEdgeCases:
    """Test edge cases and error conditions."""

    def test_inverse3_singular_matrix(self):
        """Test behavior with singular (non-invertible) matrices."""
        from brutus.utils.math import inverse3

        # Create a singular matrix (all rows identical)
        A = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=float)

        # Should still compute something (though not meaningful mathematically)
        # The function doesn't check for singularity
        result = inverse3(A)
        assert result.shape == (3, 3)
        # Note: Result will contain inf/nan values due to zero determinant

    def test_chisquare_logpdf_scalar_input(self):
        """Test chi-square log-PDF with scalar input."""
        from brutus.utils.math import chisquare_logpdf

        # Scalar input
        x = 2.0
        df = 3

        result = chisquare_logpdf(x, df)
        expected = scipy.stats.chi2.logpdf(x, df)

        np.testing.assert_almost_equal(result, expected, decimal=10)

    def test_truncnorm_pdf_scalar_input(self):
        """Test truncated normal PDF with scalar input."""
        from brutus.utils.math import truncnorm_pdf

        # Scalar input
        x = 1.0
        a, b = -2.0, 2.0

        result = truncnorm_pdf(x, a, b)
        expected = scipy.stats.truncnorm.pdf(x, a, b)

        np.testing.assert_almost_equal(result, expected, decimal=10)


@pytest.mark.integration
class TestMathIntegration:
    """Integration tests for mathematical functions."""

    def test_matrix_operations_pipeline(self):
        """Test a pipeline using multiple matrix operations."""
        from brutus.utils.math import inverse3

        np.random.seed(42)

        # Create batch of random matrices
        batch_size = 5
        A = np.random.random((batch_size, 3, 3))
        A = (
            A + np.transpose(A, (0, 2, 1)) + 3 * np.eye(3)[None, :, :]
        )  # Make invertible

        # Compute inverses
        A_inv = inverse3(A)

        # Verify A * A_inv = I for each matrix
        for i in range(batch_size):
            product = np.dot(A[i], A_inv[i])
            np.testing.assert_array_almost_equal(product, np.eye(3), decimal=10)

    def test_statistical_consistency(self):
        """Test statistical properties of distribution functions."""
        from brutus.utils.math import truncnorm_pdf, chisquare_logpdf

        # Test that truncated normal PDF integrates to 1 (approximately)
        x = np.linspace(-1.5, 1.5, 1000)
        a, b = -2.0, 2.0
        pdf_vals = truncnorm_pdf(x, a, b)

        # Numerical integration (trapezoidal rule)
        integral = np.trapz(pdf_vals, x)
        np.testing.assert_almost_equal(integral, 1.0, decimal=2)

        # Test that chi-square probabilities are reasonable
        x = np.array([1.0, 5.0, 10.0])
        df = 5
        logpdf_vals = chisquare_logpdf(x, df)

        # All should be finite (not -inf or +inf)
        assert np.all(np.isfinite(logpdf_vals))

        # Should be negative (since they're log probabilities)
        assert np.all(logpdf_vals < 0)


if __name__ == "__main__":
    pytest.main([__file__])
