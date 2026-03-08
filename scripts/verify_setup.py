#!/usr/bin/env python3
"""
WaveCommit Environment Verification Script
Tests all required imports, GPU availability, and library functionality.
"""

import sys
from typing import Dict, Any

def test_basic_imports() -> Dict[str, Any]:
    """Test all basic Python package imports."""
    results = {}
    packages = [
        'torch', 'torchaudio', 'transformers', 'captum', 'librosa',
        'soundfile', 'jiwer', 'scipy', 'numpy', 'pandas',
        'matplotlib', 'seaborn', 'tqdm', 'datasets'
    ]
    
    for pkg in packages:
        try:
            __import__(pkg)
            results[pkg] = 'OK'
        except ImportError as e:
            results[pkg] = f'FAIL: {e}'
    
    return results

def test_tensorcommitment_imports() -> Dict[str, Any]:
    """Test TensorCommitment library imports."""
    results = {}
    tc_packages = ['tensor_commitment_lib', 'terkle', 'multibranch_merkle']
    
    for pkg in tc_packages:
        try:
            __import__(pkg)
            results[pkg] = 'OK'
        except ImportError as e:
            results[pkg] = f'FAIL: {e}'
    
    # pegasus_verkle may have different import name
    try:
        import pegasus_verkle
        results['pegasus_verkle'] = 'OK'
    except ImportError:
        try:
            # Try alternate import
            from pegasus_verkle import verkle_tree
            results['pegasus_verkle'] = 'OK'
        except ImportError as e:
            results['pegasus_verkle'] = f'FAIL: {e}'
    
    return results

def test_cuda_availability() -> Dict[str, Any]:
    """Test CUDA/GPU availability."""
    import torch
    
    results = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        results['cuda_device_name'] = torch.cuda.get_device_name(0)
        results['cuda_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    
    return results

def test_hubert_load() -> Dict[str, Any]:
    """Test HuBERT model loading (without downloading full weights)."""
    from transformers import HubertModel, HubertConfig
    
    results = {}
    try:
        # Just test config loading to verify transformers works
        config = HubertConfig.from_pretrained('facebook/hubert-large-ll60k')
        results['hubert_config'] = 'OK'
        results['hidden_size'] = config.hidden_size
        results['num_layers'] = config.num_hidden_layers
    except Exception as e:
        results['hubert_config'] = f'FAIL: {e}'
    
    return results

def test_tensorcommitment_basic() -> Dict[str, Any]:
    """Test TensorCommitment basic functionality."""
    results = {}
    
    try:
        import tensor_commitment_lib
        
        # Create a small wrapper
        num_vars = 2
        degree_bound = 4
        wrapper = tensor_commitment_lib.PSTWrapper(num_vars, degree_bound)
        
        # Create dummy coefficients (4x4 = 16 coefficients for 2 vars, degree 4)
        coeffs = list(range(16))
        
        # Test commit
        commitment = wrapper.commit(coeffs)
        results['commit'] = 'OK'
        results['commitment_len'] = len(commitment)
        
        # Test evaluate
        point = [1, 2]
        eval_val = wrapper.evaluate_polynomial(coeffs, point)
        results['evaluate'] = 'OK'
        
        # Test prove
        proof = wrapper.prove(coeffs, point, eval_val)
        results['prove'] = 'OK'
        
        # Test verify
        is_valid = wrapper.verify(commitment, point, eval_val, proof)
        results['verify'] = 'OK' if is_valid else 'FAIL: verification returned False'
        
    except Exception as e:
        results['error'] = f'FAIL: {e}'
    
    return results

def test_terkle_basic() -> Dict[str, Any]:
    """Test Terkle library basic functionality."""
    results = {}
    
    try:
        import terkle
        results['import'] = 'OK'
        # Basic functionality test would go here
    except Exception as e:
        results['error'] = f'FAIL: {e}'
    
    return results

def test_captum_available() -> Dict[str, Any]:
    """Test Captum attribution methods availability."""
    results = {}
    
    try:
        from captum.attr import ShapleyValueSampling, LayerIntegratedGradients
        results['shapley'] = 'OK'
        results['layer_ig'] = 'OK'
    except ImportError as e:
        results['captum'] = f'FAIL: {e}'
    
    # Check for TracIn (may be in different location)
    try:
        from captum.attr import TracInCP
        results['tracin'] = 'OK'
    except ImportError:
        try:
            from captum.influence import TracInCP
            results['tracin'] = 'OK'
        except ImportError:
            results['tracin'] = 'Not available in this version'
    
    return results

def main():
    """Run all verification tests."""
    print("=" * 60)
    print("WaveCommit Environment Verification")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Basic imports
    print("\n1. Testing basic Python imports...")
    results = test_basic_imports()
    for pkg, status in results.items():
        symbol = '✓' if status == 'OK' else '✗'
        print(f"   {symbol} {pkg}: {status}")
        if status != 'OK':
            all_passed = False
    
    # Test 2: TensorCommitment imports
    print("\n2. Testing TensorCommitment library imports...")
    results = test_tensorcommitment_imports()
    for pkg, status in results.items():
        symbol = '✓' if status == 'OK' else '✗'
        print(f"   {symbol} {pkg}: {status}")
        if status != 'OK':
            all_passed = False
    
    # Test 3: CUDA availability
    print("\n3. Testing CUDA/GPU availability...")
    results = test_cuda_availability()
    for key, value in results.items():
        print(f"   {key}: {value}")
    
    # Test 4: HuBERT config loading
    print("\n4. Testing HuBERT configuration...")
    results = test_hubert_load()
    for key, value in results.items():
        symbol = '✓' if 'OK' in str(value) or isinstance(value, int) else '✗'
        print(f"   {symbol} {key}: {value}")
        if 'FAIL' in str(value):
            all_passed = False
    
    # Test 5: TensorCommitment functionality
    print("\n5. Testing TensorCommitment functionality...")
    results = test_tensorcommitment_basic()
    for key, value in results.items():
        symbol = '✓' if 'OK' in str(value) or isinstance(value, int) else '✗'
        print(f"   {symbol} {key}: {value}")
        if 'FAIL' in str(value):
            all_passed = False
    
    # Test 6: Terkle functionality
    print("\n6. Testing Terkle library...")
    results = test_terkle_basic()
    for key, value in results.items():
        symbol = '✓' if 'OK' in str(value) else '✗'
        print(f"   {symbol} {key}: {value}")
        if 'FAIL' in str(value):
            all_passed = False
    
    # Test 7: Captum availability
    print("\n7. Testing Captum attribution methods...")
    results = test_captum_available()
    for key, value in results.items():
        symbol = '✓' if 'OK' in str(value) else '⚠'
        print(f"   {symbol} {key}: {value}")
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All critical tests PASSED")
        print("  Environment is ready for WaveCommit development")
        return 0
    else:
        print("✗ Some tests FAILED")
        print("  Please fix the issues above before proceeding")
        return 1

if __name__ == '__main__':
    sys.exit(main())
