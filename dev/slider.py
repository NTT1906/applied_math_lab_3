import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import cv2  # OpenCV for comparison
import os

# ========== Image Read and Save Functions ==========
def read_img(filepath):
	img = Image.open(filepath).convert('RGB')
	return np.array(img, dtype=np.float32)

def save_img(img_arr, filepath):
	os.makedirs(os.path.dirname(filepath), exist_ok=True)
	Image.fromarray(img_arr).save(filepath)

# --- Contrast functions ---
def adjust_contrast_no_percentiles(img_arr, contrast_factor):
	normalized = img_arr / 255.0
	normalized -= 0.5
	normalized *= contrast_factor
	normalized += 0.5
	adjusted = np.clip(normalized * 255, 0, 255)
	return adjusted.astype(np.uint8)

def adjust_contrast_no_percentile2(img_arr, contrast_factor):
	normalized = img_arr / 255.0
	normalized *= contrast_factor
	adjusted = np.clip(normalized * 255, 0, 255)
	return adjusted.astype(np.uint8)

def adjust_contrast_with_percentiles(img_arr, contrast_factor):
	if contrast_factor == 1.0:
		return img_arr
	minval, maxval = np.percentile(img_arr, [2, 98])
	img_arr = np.clip(img_arr, minval, maxval)
	normalized = (img_arr - minval) / (maxval - minval)
	normalized -= 0.5
	normalized *= contrast_factor
	normalized += 0.5
	adjusted = np.clip(normalized * 255, 0, 255)
	return adjusted.astype(np.uint8)

def contrast_method_1(image: np.ndarray, factor: float):
	mid = 127.5
	new = (image - mid) * factor + mid
	return np.clip(new, 0, 255).astype(np.uint8)

def contrast_method_2(image: np.ndarray, factor: float):
	min_val = image.min(axis=(0, 1), keepdims=True)
	max_val = image.max(axis=(0, 1), keepdims=True)
	norm = (image - min_val) / (max_val - min_val + 1e-8)
	adjusted = (norm - 0.5) * factor + 0.5
	rescaled = adjusted * 255
	return np.clip(rescaled, 0, 255).astype(np.uint8)

def contrast_method_3(image: np.ndarray, factor: float):
	# mid = image.mean(axis=(0, 1), keepdims=True)
	mid = image.mean()
	new = (image - mid) * factor + mid
	return np.clip(new, 0, 255).astype(np.uint8)

def contrast_method_4(image: np.ndarray, factor: float):
	min_val = image.min(axis=(0, 1), keepdims=True)
	max_val = image.max(axis=(0, 1), keepdims=True)
	mid = (min_val + max_val) / 2.0
	new = (image - mid) * factor + mid
	return np.clip(new, 0, 255).astype(np.uint8)

def contrast_method_5(image, factor):
	if factor == 1.0:
		return image

	minval, maxval = np.percentile(image.reshape(-1), [2, 98])
	image = np.clip(image, minval, maxval)
	normalized = (image - minval) / (maxval - minval)

	normalized -= 0.5
	normalized *= factor
	normalized += 0.5

	adjusted = np.clip(normalized * 255, 0, 255)
	return adjusted.astype(np.uint8)

def contrast_method_6(image, factor):
	if factor == 1.0:
		return image

	flat_img = image.reshape(-1)
	minval, maxval = np.percentile(flat_img, [2, 98])
	image = np.clip(image, minval, maxval)
	normalized = (image - minval) / (maxval - minval)

	normalized -= 0.5
	normalized *= factor
	normalized += 0.5

	adjusted = np.clip(normalized * 255, 0, 255)
	return adjusted.astype(np.uint8)

def contrast_method_7(img_arr, factor):
	if factor == 1.0:
		return img_arr
	minval, maxval = np.percentile(img_arr, [2, 98])
	mid = (minval + maxval) / 2.0
	new = (img_arr - mid) * factor + mid
	return np.clip(new, 0, 255).astype(np.uint8)

def contrast_opencv(img: np.ndarray, factor: float):
	img = img.astype(np.float32)

	# Simulate OpenCV's contrast math from your controller()
	contrast = int((factor - 1.0) * 127)

	if contrast != 0:
		alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
		gamma = 127 * (1 - alpha)
		img = alpha * img + gamma

	return np.clip(img, 0, 255).astype(np.uint8)
	# adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
	# return adjusted

def img_change_contrast_noscale(img_2d: np.array, factor: float):
	img_2d = img_2d.astype(np.float32)
	img_2d[..., :3] -= 127
	img_2d[..., :3] *= factor
	img_2d[..., :3] += 127
	img_2d = np.clip(img_2d, 0, 255)
	return img_2d.astype(np.uint8)

# --- Load image once ---
original_img = Image.open("../bank.jpg")
original_array = np.array(original_img)
pil_original = Image.fromarray(original_array)

highcon_img = Image.open("../highcon.jpg")
lowcon_img = Image.open("../lowcon.jpg")

# Resize and convert to array
highcon_array = np.array(highcon_img)
lowcon_array = np.array(lowcon_img)

# --- GUI setup ---
root = tk.Tk()
root.title("Contrast Adjustment")

iw = 720
ih = 1460

iw //= 4
ih //= 4

tk_original = ImageTk.PhotoImage(pil_original.resize((iw, ih)))

# Contrast factor label
factor_label = tk.Label(root, text="Contrast Factor: 1.00", font=("Arial", 14))
factor_label.pack(pady=5)

# Slider
slider = ttk.Scale(root, from_=-2.5, to=2.5, orient="horizontal", value=0.5, length=400)
slider.pack(pady=10)

# Scrollable canvas setup
canvas = tk.Canvas(root, width=1920, height=900)
canvas.pack(side=tk.LEFT, fill="both", expand=True)

scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill="y")
canvas.configure(yscrollcommand=scrollbar.set)

img_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=img_frame, anchor="nw")

def configure_scroll_region(event):
	canvas.configure(scrollregion=canvas.bbox("all"))

img_frame.bind("<Configure>", configure_scroll_region)

def mse(imageA, imageB):
    return np.mean((imageA.astype(np.float32) - imageB.astype(np.float32)) ** 2)

def psnr(imageA, imageB):
    err = mse(imageA, imageB)
    if err == 0:
        return float("inf")
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(err))

def compare_histograms(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if img1.ndim == 3 else img1
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if img2.ndim == 3 else img2
    hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# --- Image labels in 4x3 grid ---

image_processors = [
	("Original", lambda arr, f: original_array),
	("No Percentile", adjust_contrast_no_percentiles),
	# ("No Percentile 2", adjust_contrast_no_percentile2),
	("With Percentile", adjust_contrast_with_percentiles),
	# ("Con 1", contrast_method_1),
	# ("Con 2", contrast_method_2),
	# ("Con 3", contrast_method_3),
	# ("Con 4", contrast_method_4),
	# ("Con 5", contrast_method_5),
	# ("Con 6", contrast_method_6),
	("No scale", img_change_contrast_noscale),
	# ("OpenCV", contrast_opencv),
	("High Contrast", lambda arr, f: highcon_array),
	("Low Contrast", lambda arr, f: lowcon_array),
]

# --- Create image labels dynamically ---
img_labels = []
tk_images = []  # to hold references

for idx, (label_text, _) in enumerate(image_processors):
	row = idx // 8
	col = idx % 8
	lbl = tk.Label(img_frame, text=label_text, compound="top")
	lbl.grid(row=row, column=col, padx=10, pady=10)
	img_labels.append(lbl)
	tk_images.append(None)  # placeholder

# --- Image update logic ---
def _update(factor):
	for idx, (label_text, func) in enumerate(image_processors):
		img_array = func(original_array, factor)
		pil_img = Image.fromarray(img_array).resize((iw, ih))
		tk_img = ImageTk.PhotoImage(pil_img)

		img_labels[idx].configure(image=tk_img)
		img_labels[idx].image = tk_img
		tk_images[idx] = tk_img  # prevent garbage collection

		if label_text in ("Original", "High Contrast", "Low Contrast"):
			continue  # Don't compare these to themselves
		print(f"\nSave to 'img/contrast/{label_text}_{factor}.png'")
		save_img(img_array, f'img/contrast/{label_text}_{factor}.png')
		# Resize both arrays to match
		test_img = cv2.resize(img_array, (iw, ih))
		high_img = cv2.resize(highcon_array, (iw, ih))
		low_img = cv2.resize(lowcon_array, (iw, ih))

		# Compare
		print(f"=== {label_text} ===")
		if factor == 0.5:
			print(f"→ Compared to 0.5 Low Contrast:")
			print(f"   MSE  : {mse(test_img, low_img):.2f}")
			print(f"   PSNR : {psnr(test_img, low_img):.2f}")
			print(f"   Hist : {compare_histograms(test_img, low_img):.4f}")
		elif factor == 1.5:
			print(f"→ Compared to 1.5 High Contrast:")
			print(f"   MSE  : {mse(test_img, high_img):.2f}")
			print(f"   PSNR : {psnr(test_img, high_img):.2f}")
			print(f"   Hist : {compare_histograms(test_img, high_img):.4f}")
		# for name, ref_img in [("High Contrast", high_img), ("Low Contrast", low_img)]:
		# 	print(f"→ Compared to {name}:")
		# 	print(f"   MSE  : {mse(test_img, ref_img):.2f}")
		# 	print(f"   PSNR : {psnr(test_img, ref_img):.2f}")
		# 	print(f"   Hist : {compare_histograms(test_img, ref_img):.4f}")

def update_images(event=None):
	factor = slider.get()
	factor_label.config(text=f"Contrast Factor: {factor:.2f}")
	_update(factor)

# Initial image load
_update(0.5)
_update(1.5)
# update_images()

# Slider binding
slider.bind("<Motion>", update_images)
slider.bind("<ButtonRelease-1>", update_images)

# Run the GUI
root.mainloop()

# test: 0.5 -> loai con 4, no pep 2
# test: 1.5 -> loai con 2