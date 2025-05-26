#!/usr/bin/env python3
"""
Test script to verify the cleaned pipeline works correctly.

This script performs a quick validation of the main components
without running the full pipeline.
"""

import sys
from pathlib import Path

# # Add src to Python path
# sys.path.append('src')

def test_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing imports...")
    
    try:
        # Configuration
        from config.base import HORIZON, FREQUENCY, SEED
        print("  ✅ Configuration imports successful")
        
        # Model registry
        from src.models.model_registry import ModelRegistry
        print("  ✅ Model registry import successful")
        
        # Data preparation
        from src.dataset.data_preparation import prepare_data
        print("  ✅ Data preparation import successful")
        
        # HPO
        from src.hpo.hyperparameter_tuning import run_complete_hpo_pipeline
        print("  ✅ HPO import successful")
        
        # Utils
        from src.utils.other import setup_environment
        print("  ✅ Utils import successful")
        
        print("  🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_model_registry():
    """Test the model registry functionality."""
    print("\n🧪 Testing model registry...")
    
    try:
        from src.models.model_registry import ModelRegistry
        
        # Test model summary
        summary = ModelRegistry.get_model_summary()
        print(f"  ✅ Model summary: {summary}")
        
        # Test statistical models
        stat_models_full = ModelRegistry.get_statistical_models()
        print(f"  ✅ Statistical models - Fast: {len(stat_models_fast)}, Full: {len(stat_models_full)}")
        
        # Test neural models)
        neural_models_full = ModelRegistry.get_neural_models(7)
        print(f"  ✅ Neural models - Fast: {len(neural_models_fast)}, Full: {len(neural_models_full)}")
        
        # Test auto models
        auto_models_full = ModelRegistry.get_auto_models(7, num_samples=2)
        print(f"  ✅ Auto models - Fast: {len(auto_models_fast)}, Full: {len(auto_models_full)}")
        
        # Test loss functions
        loss_functions = ModelRegistry.get_loss_functions()
        print(f"  ✅ Loss functions: {list(loss_functions.keys())}")
        
        print("  🎉 Model registry tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Model registry test error: {e}")
        return False

def test_configuration():
    """Test configuration values."""
    print("\n🧪 Testing configuration...")
    
    try:
        from config.base import (
            HORIZON, FREQUENCY, SEED, DATA_PATH, RESULTS_DIR,
            HPO_DIR, CV_DIR, FINAL_DIR
        )
        
        print(f"  ✅ Horizon: {HORIZON}")
        print(f"  ✅ Frequency: {FREQUENCY}")
        print(f"  ✅ Seed: {SEED}")
        print(f"  ✅ Data path: {DATA_PATH}")
        print(f"  ✅ Results dir: {RESULTS_DIR}")
        
        # Check if directories exist or can be created
        from pathlib import Path
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        Path(HPO_DIR).mkdir(parents=True, exist_ok=True)
        Path(CV_DIR).mkdir(parents=True, exist_ok=True)
        Path(FINAL_DIR).mkdir(parents=True, exist_ok=True)
        
        print("  ✅ Output directories created successfully")
        print("  🎉 Configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test error: {e}")
        return False

def test_data_availability():
    """Test if data file exists."""
    print("\n🧪 Testing data availability...")
    
    try:
        from config.base import DATA_PATH
        from pathlib import Path
        
        data_file = Path(DATA_PATH)
        if data_file.exists():
            print(f"  ✅ Data file found: {DATA_PATH}")
            print(f"  ✅ File size: {data_file.stat().st_size / 1024 / 1024:.2f} MB")
            return True
        else:
            print(f"  ⚠️  Data file not found: {DATA_PATH}")
            print("     This is expected if you haven't prepared the data yet.")
            return False
            
    except Exception as e:
        print(f"  ❌ Data availability test error: {e}")
        return False

def test_pipeline_initialization():
    """Test pipeline initialization."""
    print("\n🧪 Testing pipeline initialization...")
    
    try:
        # Import main pipeline
        sys.path.append('.')
        from main_clean import BitcoinForecastingPipeline
        
        # Test initialization
        
        pipeline_full = BitcoinForecastingPipeline(skip_hpo=False)
        print("  ✅ Full mode pipeline initialized")
        
        print("  🎉 Pipeline initialization tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Pipeline initialization test error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("🚀 BITCOIN FORECASTING PIPELINE - VALIDATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Model Registry", test_model_registry),
        ("Data Availability", test_data_availability),
        ("Pipeline Initialization", test_pipeline_initialization),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        result = test_func()
        if result:
            passed += 1
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Ensure your data is in the correct location")
        print("2. Run: python main_clean.py --fast-mode --skip-hpo (for quick test)")
        print("3. Run: python main_clean.py (for full pipeline)")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        
    print("=" * 60)

if __name__ == "__main__":
    main()
