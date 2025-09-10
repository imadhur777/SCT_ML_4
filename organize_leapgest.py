import os
import shutil
import glob
from sklearn.model_selection import train_test_split

# ✅ Correct dataset path
data_dir = r"C:\Users\amrit\OneDrive\Desktop\SkillSoft4\leapGestRecog"

# New train/test output paths
output_train = r"C:\Users\amrit\OneDrive\Desktop\SkillSoft4\Train"
output_test  = r"C:\Users\amrit\OneDrive\Desktop\SkillSoft4\Test"

os.makedirs(output_train, exist_ok=True)
os.makedirs(output_test, exist_ok=True)

# Gesture class names in LeapGestRecog
gestures = [
    "01_palm", "02_l", "03_fist", "04_fist_moved",
    "05_thumb", "06_index", "07_ok", "08_palm_moved",
    "09_c", "10_down"
]

# Process each gesture
for gesture in gestures:
    print(f"Processing {gesture}...")

    # Match LeapGestRecog structure: subject -> gesture -> frames
    pattern = os.path.join(data_dir, "*", gesture, "*.png")
    files = glob.glob(pattern)

    if len(files) == 0:
        print(f"⚠️ No images found for {gesture}")
        continue

    print(f"Found {len(files)} files for {gesture}")

    # Split into train/test
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

    # Create class folders
    train_gesture_dir = os.path.join(output_train, gesture)
    test_gesture_dir  = os.path.join(output_test, gesture)
    os.makedirs(train_gesture_dir, exist_ok=True)
    os.makedirs(test_gesture_dir, exist_ok=True)

    # Copy files with subject prefix to avoid overwriting
    for f in train_files:
        subject = os.path.basename(os.path.dirname(os.path.dirname(f)))  # e.g. "00"
        fname = subject + "_" + os.path.basename(f)
        shutil.copy(f, os.path.join(train_gesture_dir, fname))

    for f in test_files:
        subject = os.path.basename(os.path.dirname(os.path.dirname(f)))  # e.g. "00"
        fname = subject + "_" + os.path.basename(f)
        shutil.copy(f, os.path.join(test_gesture_dir, fname))

print("✅ LeapGestRecog reorganized into Train/ and Test/ folders.")
