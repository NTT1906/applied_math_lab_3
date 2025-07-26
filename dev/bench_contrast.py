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

def img_change_contrast1(img_2d: np.array, factor: float):
	norm = img_2d / 255.0
	norm -= 0.5
	norm *= factor
	norm += 0.5
	return np.clip(norm * 255, 0, 255).astype(np.uint8)

def img_change_contrast2(img_2d: np.array, factor: float):
	img_2d = img_2d.astype(np.float32)
	img_2d[..., :3] -= 127
	img_2d[..., :3] *= factor
	img_2d[..., :3] += 127
	img_2d = np.clip(img_2d, 0, 255)
	return img_2d.astype(np.uint8)

def img_change_contrast2b(img_2d: np.array, factor: float):
	img_2d = img_2d.astype(np.float32)
	img_2d[..., :3] -= 127
	img_2d[..., :3] *= factor
	img_2d[..., :3] += 127
	return np.clip(img_2d, 0, 255).astype(np.uint8)

def img_change_contrast3(img_2d: np.array, factor: float):
    img_2d = img_2d.astype(np.float32)
    rgb_mask = img_2d[:, :, :3]
    rgb_mask -= 127.0
    rgb_mask *= factor
    rgb_mask += 127.0
    np.clip(rgb_mask, 0, 255, out=rgb_mask)
    img_2d[:, :, :3] = rgb_mask
    return img_2d.astype(np.uint8)

def img_change_contrast4(img_2d: np.array, factor: float) -> np.array:
	img_2d = img_2d.astype(np.float32)
	rgb_mask = img_2d[:, :, :3] # only modify the first three channel rgb and keep alpha intact
	rgb_mask -= 127.0  # map the range from (0,256) to (-127, 128) with dark parts being lower than 0 and the opposite for light part
	rgb_mask *= factor # distance the dark and light part by factor amount
	rgb_mask += 127.0  # remap it back to normal range (0, 256)
	np.clip(rgb_mask, 0, 255, out=rgb_mask) # clip out overflowed value
	return img_2d.astype(np.uint8)

# ========== Benchmark Setup ==========

img = read_img('../bank.jpg')
contrast_factor = 1.5
num_trials = 2000

funcs = [
	# ('0. No Percentiles', lambda: adjust_contrast_no_percentiles(img, contrast_factor)),
	# ('1. No Percentile 2', lambda: adjust_contrast_no_percentile2(img, contrast_factor)),
	# ('2. With Percentiles', lambda: adjust_contrast_with_percentiles(img, contrast_factor)),
	# ('3. Method 1 (Fixed Mid)', lambda: contrast_method_1(img, contrast_factor)),
	# ('4. Method 2 (MinMax Norm)', lambda: contrast_method_2(img, contrast_factor)),
	# ('5. Method 3 (Mean Mid)', lambda: contrast_method_3(img, contrast_factor)),
	# ('6. Method 4 (Dynamic Mid)', lambda: contrast_method_4(img, contrast_factor)),
	# ('7. Method 5 (Modded 3)', lambda: contrast_method_5(img, contrast_factor)),
	# ('8. Method 6 (Modded 5)', lambda: contrast_method_6(img, contrast_factor)),
	# ('9. Method 7 (Modded 4 combine with 2)', lambda: contrast_method_7(img, contrast_factor)),
	# ('1. Norm float', lambda: img_change_contrast1(img, contrast_factor)),
	# ('2. Raw float 2 ', lambda: img_change_contrast2(img, contrast_factor)),
	# ('2. Raw float 2b', lambda: img_change_contrast2(img, contrast_factor)),
	('2. Raw float 3 ', lambda: img_change_contrast3(img, contrast_factor)),
	('2. Raw float 4 ', lambda: img_change_contrast3(img, contrast_factor)),
]

import tracemalloc
for name, func in funcs:
	tracemalloc.start()
	temp = func()
	current, peak = tracemalloc.get_traced_memory()
	tracemalloc.stop()
	print(name, current, peak, temp.shape, temp.dtype)
	print(f"Save to 'img/contrast/{name}.png'")
	save_img(temp, 'img/contrast/' + name + '.png')

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

save_img(img.astype(np.uint8), 'img/contrast/original.png')

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
