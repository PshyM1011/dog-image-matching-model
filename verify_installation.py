"""
Verification script to check if all dependencies are installed correctly.
Run this after installing requirements.txt to verify your setup.
"""
import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ {package_name or module_name} - OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name} - MISSING")
        print(f"   Error: {e}")
        return False

def main():
    print("=" * 60)
    print("Verifying Installation")
    print("=" * 60)
    print()
    
    checks = []
    
    # Core ML frameworks
    print("Core ML Frameworks:")
    checks.append(check_import("torch", "PyTorch"))
    checks.append(check_import("torchvision", "torchvision"))
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("   CUDA not available (CPU mode)")
    except:
        pass
    
    print()
    print("Computer Vision:")
    checks.append(check_import("cv2", "opencv-python"))
    checks.append(check_import("PIL", "Pillow"))
    
    print()
    print("ML Utilities:")
    checks.append(check_import("numpy", "numpy"))
    checks.append(check_import("sklearn", "scikit-learn"))
    checks.append(check_import("matplotlib", "matplotlib"))
    
    print()
    print("Similarity Search:")
    checks.append(check_import("faiss", "faiss-cpu/faiss-gpu"))
    
    print()
    print("Object Detection:")
    checks.append(check_import("ultralytics", "ultralytics"))
    
    print()
    print("Utilities:")
    checks.append(check_import("tqdm", "tqdm"))
    
    print()
    print("=" * 60)
    
    # Check project structure
    print("\nChecking Project Structure:")
    import os
    required_dirs = ['src', 'data', 'src/detector', 'src/preprocessing', 'src/model', 'src/utils']
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/ - OK")
        else:
            print(f"❌ {dir_path}/ - MISSING")
            checks.append(False)
    
    print()
    print("=" * 60)
    
    # Check project modules
    print("\nChecking Project Modules:")
    try:
        from src.model import DualViewFusionModel
        print("✅ src.model - OK")
    except Exception as e:
        print(f"❌ src.model - ERROR: {e}")
        checks.append(False)
    
    try:
        from src.preprocessing import get_train_transforms
        print("✅ src.preprocessing - OK")
    except Exception as e:
        print(f"❌ src.preprocessing - ERROR: {e}")
        checks.append(False)
    
    try:
        from src.detector import DogDetector
        print("✅ src.detector - OK")
    except Exception as e:
        print(f"❌ src.detector - ERROR: {e}")
        checks.append(False)
    
    try:
        from src.utils import DualViewDataset
        print("✅ src.utils - OK")
    except Exception as e:
        print(f"❌ src.utils - ERROR: {e}")
        checks.append(False)
    
    print()
    print("=" * 60)
    
    # Summary
    all_passed = all(checks)
    if all_passed:
        print("\n🎉 All checks passed! Your installation is ready.")
        print("\nNext steps:")
        print("1. Organize your dataset in data/train/, data/val/, data/test/")
        print("2. Run: python src/train.py --data_dir data")
        print("3. See QUICKSTART.md for detailed instructions")
    else:
        print("\n⚠️  Some checks failed. Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

if __name__ == '__main__':
    main()

