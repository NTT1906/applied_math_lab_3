import numpy as np
import timeit
import matplotlib.pyplot as plt
from PIL import Image
import os

# ========== Image Read and Save Functions ==========
def read_img(filepath):
	img = Image.open(filepath).convert('RGB')
	return np.array(img, dtype=np.float32)

def save_img(img_arr, filepath):
	os.makedirs(os.path.dirname(filepath), exist_ok=True)
	Image.fromarray(img_arr).save(filepath)

# ========== Contrast Functions ==========

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

def contrast_method_1(image, factor):
	mid = 127.5
	new = (image - mid) * factor + mid
	return np.clip(new, 0, 255).astype(np.uint8)

def contrast_method_2(image, factor):
	min_val = image.min(axis=(0, 1), keepdims=True)
	max_val = image.max(axis=(0, 1), keepdims=True)
	norm = (image - min_val) / (max_val - min_val + 1e-8)
	adjusted = (norm - 0.5) * factor + 0.5
	rescaled = adjusted * 255
	return np.clip(rescaled, 0, 255).astype(np.uint8)

def contrast_method_3(image, factor):
	# mid = image.mean(axis=(0, 1), keepdims=True)
	mid = img.mean()
	new = (image - mid) * factor + mid
	return np.clip(new, 0, 255).astype(np.uint8)

def contrast_method_4(image, factor):
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

# ========== Benchmark Setup ==========

img = read_img('../cat2.jpg')
mid = img.mean(axis=(0, 1), keepdims=True)
print(mid)
mid = img.mean()
print(mid)
contrast_factor = 1.5
num_trials = 50

funcs = [
	('0. No Percentiles', lambda: adjust_contrast_no_percentiles(img, contrast_factor)),
	('1. No Percentile 2', lambda: adjust_contrast_no_percentile2(img, contrast_factor)),
	('2. With Percentiles', lambda: adjust_contrast_with_percentiles(img, contrast_factor)),
	('3. Method 1 (Fixed Mid)', lambda: contrast_method_1(img, contrast_factor)),
	('4. Method 2 (MinMax Norm)', lambda: contrast_method_2(img, contrast_factor)),
	('5. Method 3 (Mean Mid)', lambda: contrast_method_3(img, contrast_factor)),
	('6. Method 4 (Dynamic Mid)', lambda: contrast_method_4(img, contrast_factor)),
	('7. Method 5 (Modded 3)', lambda: contrast_method_5(img, contrast_factor)),
	('8. Method 6 (Modded 5)', lambda: contrast_method_6(img, contrast_factor)),
	('9. Method 7 (Modded 4 combine with 2)', lambda: contrast_method_7(img, contrast_factor)),
]


# for name, func in funcs.items():
# 	save_img(func(), 'img/' + name + '.png')

for name, func in funcs:
	func()
import random
# ========== Benchmarking ==========
results = {name: [] for name, _ in funcs}
amount = 200
for _ in range(amount):
	random.shuffle(funcs)
	for name, func in funcs:
		t = timeit.timeit(func, number=1)
		results[name].append(t)
# ========== Plotting ==========
plt.figure(figsize=(12, 7))
for name in results:
	avg = sum(results[name]) / len(results[name])
	print(f"{name}: {avg:.8f} seconds per run")

i = 0
for name in results:
	plt.plot(results[name], label=name)
	# Add label near the start of each line
	plt.text(x=0.1, y=results[name][0], s=str(i), fontsize=8, verticalalignment='bottom')
	i += 1
plt.xlabel('Trial')
plt.ylabel('Time (seconds)')
plt.title('Contrast Adjustment Benchmark Comparison')
plt.legend()

plt.grid(True)
plt.tight_layout()
plt.show()
