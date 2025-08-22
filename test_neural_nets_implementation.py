#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify neural networks module refactoring preserves functionality.

This script compares outputs from the original seds.py neural network classes
with the new core.neural_nets module to ensure identical functionality.

Run this script from the project root directory to verify the refactoring
was successful before proceeding with the rest of the core module reorganization.
"""

import sys
import numpy as np
import warnings

from brutus.data import filters

# Add source directory to path for imports
sys.path.insert(0, "src")


def test_neural_nets_implementation():
    """
    Test that FastNN and FastNNPredictor produce identical outputs
    between original and new implementations.
    """
    print("=" * 60)
    print("Testing Neural Networks Module Implementation")
    print("=" * 60)

    try:
        # Import original classes from seds
        from brutus.seds import FastNN as OriginalFastNN
        from brutus.seds import FastNNPredictor as OriginalFastNNPredictor

        print("✓ Successfully imported original neural network classes")
    except ImportError as e:
        print(f"✗ Failed to import original classes: {e}")
        return False

    try:
        # Import new classes from core.neural_nets
        from brutus.core.neural_nets import FastNN as NewFastNN
        from brutus.core.neural_nets import FastNNPredictor as NewFastNNPredictor

        print("✓ Successfully imported new neural network classes")
    except ImportError as e:
        print(f"✗ Failed to import new classes: {e}")
        return False

    # Test parameters
    test_filters = filters.ps  # Use PanSTARRS filters for testing
    test_params = [
        # [log10(Teff), log g, [Fe/H], [α/Fe], Av, Rv]
        [3.76, 4.44, 0.0, 0.0, 0.0, 3.1],  # Solar analog, no extinction
        [3.60, 2.5, -0.5, 0.3, 0.5, 3.3],  # Red giant with extinction
        [4.0, 4.0, -1.0, 0.4, 1.0, 2.5],  # Hot star, low metallicity
        [3.5, 5.0, 0.2, 0.0, 0.1, 4.0],  # Cool dwarf, high metallicity
        [3.9, 3.5, -2.0, 0.6, 2.0, 3.8],  # Evolved star, very metal-poor
    ]

    # Test SED parameters
    sed_test_cases = [
        {"logt": 3.76, "logg": 4.44, "feh_surf": 0.0, "logl": 0.0, "dist": 1000.0},
        {
            "logt": 3.60,
            "logg": 2.5,
            "feh_surf": -0.5,
            "logl": 1.5,
            "av": 0.5,
            "dist": 2000.0,
        },
        {
            "logt": 4.0,
            "logg": 4.0,
            "feh_surf": -1.0,
            "logl": -0.5,
            "av": 1.0,
            "rv": 2.5,
            "dist": 500.0,
        },
    ]

    print(f"\nTesting with {len(test_filters)} filters: {test_filters}")
    print(f"Using {len(test_params)} parameter test cases")

    # Create mock neural network file path (will use default if exists)
    nn_file = None  # Use default for both implementations

    try:
        # Initialize both versions with same parameters
        print("\nInitializing neural network classes...")

        # Suppress verbose output during testing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            print("1")
            original_nn = OriginalFastNN(
                filters=test_filters, nnfile=nn_file, verbose=False
            )
            print("2")
            new_nn = NewFastNN(filters=test_filters, nnfile=nn_file, verbose=False)
            print("3")

            original_predictor = OriginalFastNNPredictor(
                filters=test_filters, nnfile=nn_file, verbose=False
            )
            new_predictor = NewFastNNPredictor(
                filters=test_filters, nnfile=nn_file, verbose=False
            )

        print("✓ Successfully initialized both implementations")

    except Exception as e:
        print(f"✗ Failed to initialize neural networks: {e}")
        print("Note: This may be due to missing neural network data files.")
        print("The neural network files are downloaded automatically when needed.")
        return False

    # Test 1: Compare neural network architecture
    print("\n" + "-" * 40)
    print("Test 1: Neural Network Architecture")
    print("-" * 40)

    architecture_tests = [
        ("w1", "Weight matrix 1"),
        ("w2", "Weight matrix 2"),
        ("w3", "Weight matrix 3"),
        ("b1", "Bias vector 1"),
        ("b2", "Bias vector 2"),
        ("b3", "Bias vector 3"),
        ("xmin", "Input minimum values"),
        ("xmax", "Input maximum values"),
        ("xspan", "Input parameter spans"),
    ]

    for attr, description in architecture_tests:
        try:
            orig_val = getattr(original_nn, attr)
            new_val = getattr(new_nn, attr)

            if np.allclose(orig_val, new_val, rtol=1e-12, atol=1e-12):
                print(f"✓ {description} match exactly")
            else:
                print(f"✗ {description} differ!")
                print(f"  Max difference: {np.max(np.abs(orig_val - new_val))}")
                return False

        except Exception as e:
            print(f"✗ Error comparing {description}: {e}")
            return False

    # Test 2: Compare individual method outputs
    print("\n" + "-" * 40)
    print("Test 2: Individual Method Outputs")
    print("-" * 40)

    for i, params in enumerate(test_params):
        try:
            # Test encode method
            orig_encoded = original_nn.encode(np.array(params))
            new_encoded = new_nn.encode(np.array(params))

            if np.allclose(orig_encoded, new_encoded, rtol=1e-12, atol=1e-12):
                print(f"✓ encode() method - test case {i+1}")
            else:
                print(f"✗ encode() method differs - test case {i+1}")
                return False

            # Test sigmoid method
            test_input = np.random.randn(10)
            orig_sigmoid = original_nn.sigmoid(test_input)
            new_sigmoid = new_nn.sigmoid(test_input)

            if np.allclose(orig_sigmoid, new_sigmoid, rtol=1e-12, atol=1e-12):
                print(f"✓ sigmoid() method - test case {i+1}")
            else:
                print(f"✗ sigmoid() method differs - test case {i+1}")
                return False

            # Test nneval method
            orig_nneval = original_nn.nneval(np.array(params))
            new_nneval = new_nn.nneval(np.array(params))

            if np.allclose(orig_nneval, new_nneval, rtol=1e-12, atol=1e-12):
                print(f"✓ nneval() method - test case {i+1}")
            else:
                print(f"✗ nneval() method differs - test case {i+1}")
                print(f"  Max difference: {np.max(np.abs(orig_nneval - new_nneval))}")
                return False

        except Exception as e:
            print(f"✗ Error in method testing for case {i+1}: {e}")
            return False

    # Test 3: Compare FastNNPredictor attributes
    print("\n" + "-" * 40)
    print("Test 3: FastNNPredictor Attributes")
    print("-" * 40)

    predictor_attrs = [
        ("filters", "Filter list"),
        ("NFILT", "Number of filters"),
    ]

    for attr, description in predictor_attrs:
        try:
            orig_val = getattr(original_predictor, attr)
            new_val = getattr(new_predictor, attr)

            if isinstance(orig_val, np.ndarray):
                match = np.array_equal(orig_val, new_val)
            else:
                match = orig_val == new_val

            if match:
                print(f"✓ {description} match")
            else:
                print(f"✗ {description} differ!")
                print(f"  Original: {orig_val}")
                print(f"  New: {new_val}")
                return False

        except Exception as e:
            print(f"✗ Error comparing {description}: {e}")
            return False

    # Test 4: Compare SED predictions
    print("\n" + "-" * 40)
    print("Test 4: SED Prediction Outputs")
    print("-" * 40)

    for i, test_case in enumerate(sed_test_cases):
        try:
            orig_sed = original_predictor.sed(**test_case)
            new_sed = new_predictor.sed(**test_case)

            if np.allclose(orig_sed, new_sed, rtol=1e-12, atol=1e-12):
                print(f"✓ SED prediction - test case {i+1}")
                print(f"  Parameters: {test_case}")
                print(f"  Sample magnitudes: {new_sed[:3]}")
            else:
                print(f"✗ SED prediction differs - test case {i+1}")
                print(f"  Max difference: {np.max(np.abs(orig_sed - new_sed))}")
                print(f"  Original: {orig_sed}")
                print(f"  New: {new_sed}")
                return False

        except Exception as e:
            print(f"✗ Error in SED prediction for case {i+1}: {e}")
            return False

    # Test 5: Edge cases and boundary conditions
    print("\n" + "-" * 40)
    print("Test 5: Edge Cases and Boundary Conditions")
    print("-" * 40)

    edge_cases = [
        # Test out-of-bounds parameters (should return NaN)
        {
            "logt": 2.0,
            "logg": 4.0,
            "feh_surf": 0.0,
            "logl": 0.0,
            "dist": 1000.0,
        },  # Too cool
        {
            "logt": 5.0,
            "logg": 4.0,
            "feh_surf": 0.0,
            "logl": 0.0,
            "dist": 1000.0,
        },  # Too hot
        # Test filter indexing
        {
            "logt": 3.8,
            "logg": 4.4,
            "feh_surf": 0.0,
            "logl": 0.0,
            "dist": 1000.0,
            "filt_idxs": [0, 2],
        },
        {
            "logt": 3.8,
            "logg": 4.4,
            "feh_surf": 0.0,
            "logl": 0.0,
            "dist": 1000.0,
            "filt_idxs": slice(1, 3),
        },
    ]

    for i, edge_case in enumerate(edge_cases):
        try:
            orig_sed = original_predictor.sed(**edge_case)
            new_sed = new_predictor.sed(**edge_case)

            # Handle NaN comparison
            if np.isnan(orig_sed).all() and np.isnan(new_sed).all():
                print(f"✓ Edge case {i+1} - both return NaN as expected")
            elif np.allclose(orig_sed, new_sed, rtol=1e-12, atol=1e-12, equal_nan=True):
                print(f"✓ Edge case {i+1} - outputs match")
            else:
                print(f"✗ Edge case {i+1} - outputs differ")
                print(f"  Original: {orig_sed}")
                print(f"  New: {new_sed}")
                return False

        except Exception as e:
            print(f"✗ Error in edge case {i+1}: {e}")
            return False

    # Summary
    print("\n" + "=" * 60)
    print("NEURAL NETWORKS MODULE TEST SUMMARY")
    print("=" * 60)
    print("✓ All tests passed!")
    print("✓ Neural network architecture matches exactly")
    print("✓ All methods produce identical outputs")
    print("✓ SED predictions are numerically identical")
    print("✓ Edge cases handled consistently")
    print("\nThe neural networks module refactoring was successful.")
    print("You can safely proceed with the next phase of core module reorganization.")

    return True


if __name__ == "__main__":
    """
    Run the neural networks implementation test.

    Usage:
        python test_neural_nets_implementation.py

    Exit codes:
        0: All tests passed
        1: Tests failed or error occurred
    """

    try:
        success = test_neural_nets_implementation()
        if success:
            print(f"\n🎉 All neural networks tests passed!")
            sys.exit(0)
        else:
            print(f"\n❌ Neural networks tests failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n\nTest interrupted by user.")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
