"""Test that DualViewDataset now correctly finds .JPEG files."""
import sys
from pathlib import Path
sys.path.insert(0, '.')

from src.utils import DualViewDataset

print("Testing DualViewDataset with fixed image loading...")
print("=" * 60)

# Test validation dataset
val_ds = DualViewDataset('data/val')
print(f'\nVal samples: {len(val_ds)}')

# Test test dataset
test_ds = DualViewDataset('data/test')
print(f'Test samples: {len(test_ds)}')

# Check dog4 in val
print(f'\nChecking dog4 in val:')
dog4_samples = [s for s in val_ds.samples if s['dog_id'] == 'dog4']
print(f'  Found {len(dog4_samples)} samples')
for s in dog4_samples:
    frontal_name = Path(s['frontal_path']).name
    lateral_name = Path(s['lateral_path']).name
    print(f'    - {frontal_name} + {lateral_name}')

# Check dog4 in test
print(f'\nChecking dog4 in test:')
dog4_test_samples = [s for s in test_ds.samples if s['dog_id'] == 'dog4']
print(f'  Found {len(dog4_test_samples)} samples')
for s in dog4_test_samples:
    frontal_name = Path(s['frontal_path']).name
    lateral_name = Path(s['lateral_path']).name
    print(f'    - {frontal_name} + {lateral_name}')

# Check dog3 in test
print(f'\nChecking dog3 in test:')
dog3_test_samples = [s for s in test_ds.samples if s['dog_id'] == 'dog3']
print(f'  Found {len(dog3_test_samples)} samples')
for s in dog3_test_samples:
    frontal_name = Path(s['frontal_path']).name
    lateral_name = Path(s['lateral_path']).name
    print(f'    - {frontal_name} + {lateral_name}')

print("\n" + "=" * 60)
print("If dog4 and dog3 show 1 sample each with .JPEG files, the fix worked!")




