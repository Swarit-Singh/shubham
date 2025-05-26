import numpy as np
import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
from tqdm import tqdm  # for progress bars

# --- Configuration ---
PATCH_SIZE = 5
IMAGE_DIR = r"D:\College\Btech Project\FINAL\attacks\prototype23\training_images_grayscale"
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Sampling & external memory parameters
MAX_IMAGES = 200                      # Number of images to process
SAMPLES_PER_IMAGE = 10000            # Random patches per image
LIBSVM_TRAIN_PATH = 'xgb_train.data' # Will be written in LibSVM format
LIBSVM_VAL_PATH   = 'xgb_val.data'

# XGBoost hyperparameters
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED,
    'verbosity': 1,
    # 'tree_method': 'gpu_hist'  # Uncomment if you have a CUDA GPU
}

# --- Helpers ---
def to_libsvm_line(label, features):
    """Convert a (label, feature vector) into a LibSVM-formatted line."""
    entries = [f"{i+1}:{int(val)}" for i, val in enumerate(features)]
    return f"{int(label)} " + " ".join(entries) + "\n"

def sample_random_patches_from_image(image_path, patch_size, samples_per_image):
    """
    Load a grayscale image and randomly sample up to `samples_per_image` patches.
    Returns two lists: [patch_vectors], [center_labels].
    """
    try:
        img = Image.open(image_path).convert('L')
        img_np = np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"Skipping {image_path}: {e}")
        return [], []

    rows, cols = img_np.shape
    half = patch_size // 2
    padded = np.pad(img_np, half, mode='edge')
    total = rows * cols

    if total <= samples_per_image:
        coords = [(r, c) for r in range(rows) for c in range(cols)]
    else:
        idxs = np.random.choice(total, size=samples_per_image, replace=False)
        coords = [(i // cols, i % cols) for i in idxs]

    patches, labels = [], []
    for r, c in coords:
        patch = padded[r : r + patch_size, c : c + patch_size]
        patches.append(patch.flatten())
        labels.append(img_np[r, c])

    return patches, labels

# --- Gather & Split Image Paths ---
image_extensions = ('*.png','*.jpg','*.jpeg','*.bmp','*.tif','*.tiff')
all_files = []
for ext in image_extensions:
    all_files += glob.glob(os.path.join(IMAGE_DIR, ext))
all_files = sorted(all_files)[:MAX_IMAGES]

if not all_files:
    raise FileNotFoundError(f"No images found in '{IMAGE_DIR}'")

train_files, val_files = train_test_split(all_files, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
print(f"Using {len(train_files)} train images and {len(val_files)} val images.")

# --- Write LibSVM Files for External Memory ---
print("Writing LibSVM files...")
with open(LIBSVM_TRAIN_PATH, 'w') as f_tr, open(LIBSVM_VAL_PATH, 'w') as f_val:
    for img_path in tqdm(train_files, desc="Train Images"):
        patches, labels = sample_random_patches_from_image(img_path, PATCH_SIZE, SAMPLES_PER_IMAGE)
        for p, l in zip(patches, labels):
            f_tr.write(to_libsvm_line(l, p))
    for img_path in tqdm(val_files, desc="Validation Images"):
        patches, labels = sample_random_patches_from_image(img_path, PATCH_SIZE, SAMPLES_PER_IMAGE)
        for p, l in zip(patches, labels):
            f_val.write(to_libsvm_line(l, p))
print("LibSVM files written.")

# --- Load DMatrix with External Memory & Caching ---
# Note: '?format=libsvm' tells XGBoost it's LibSVM format, 
# and '#<cache_file>' enables disk caching.
dtrain = xgb.DMatrix(f"{LIBSVM_TRAIN_PATH}?format=libsvm#xgb_train.cache")
dval   = xgb.DMatrix(f"{LIBSVM_VAL_PATH}?format=libsvm#xgb_val.cache")
print("DMatrix objects created (external memory).")

# --- Train with Early Stopping ---
evals = [(dtrain, 'train'), (dval, 'validation')]
print("Starting training...")
model = xgb.train(
    params=XGB_PARAMS,
    dtrain=dtrain,
    num_boost_round=200,
    evals=evals,
    early_stopping_rounds=10,
    verbose_eval=True
)
print("Training complete.")

# --- Save Model ---
joblib.dump(model, 'xgb_pixel_predictor.pkl')
print("Model saved to 'xgb_pixel_predictor.pkl'.")
