import os
import numpy as np
from PIL import Image
from watermark_operations import embed_pred_error, extract_pred_error

folder = "ml_training_data"
payload = "01" * 1000  # Adjust as needed

def attack_gaussian(img):
    sigma = 10
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def attack_salt_pepper(img, amount=0.02):
    noisy = img.copy()
    num_salt = np.ceil(amount * img.size * 0.5)
    # Generate random coordinates for salt
    coords = [np.random.randint(0, img.shape[axis], int(num_salt)) for axis in range(img.ndim)]
    noisy[tuple(coords)] = 255
    num_pepper = np.ceil(amount * img.size * 0.5)
    # Generate random coordinates for pepper
    coords = [np.random.randint(0, img.shape[axis], int(num_pepper)) for axis in range(img.ndim)]
    noisy[tuple(coords)] = 0
    return noisy


def attack_jpeg(img):
    pil_img = Image.fromarray(img.astype(np.uint8))
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=70)
    buf.seek(0)
    return np.array(Image.open(buf))

def attack_median(img):
    from scipy.signal import medfilt2d
    return medfilt2d(img, kernel_size=3).astype(np.uint8)

attack_funcs = [attack_gaussian, attack_salt_pepper, attack_jpeg, attack_median]

for fname in os.listdir(folder):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(folder, fname)
        img = np.array(Image.open(img_path).convert("L"))
        best_acc = -1
        best_T = 1
        for T in range(1, 6):
            wm_img, _ = embed_pred_error(img, payload, T)
            accs = []
            for attack in attack_funcs:
                attacked_img = attack(wm_img)
                ext_bits, _ = extract_pred_error(attacked_img, len(payload), T)
                matches = sum(1 for a, b in zip(payload, ext_bits) if a == b)
                acc = matches / len(payload)
                accs.append(acc)
            avg_acc = np.mean(accs)
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_T = T
        label_path = os.path.join(folder, fname.rsplit('.',1)[0]+".txt")
        with open(label_path, "w") as f:
            f.write(str(best_T))
print("Labels updated for best robustness across all attacks.")
