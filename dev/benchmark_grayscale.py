import random
import numpy as np
from time import perf_counter
import tracemalloc
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

# def change_grayscale_flat(img_2d, way='weight'):
# 	row, col, ch = img_2d.shape
# 	img_flat = img_2d.reshape(-1, ch)
#
# 	if way == 'average':
# 		gray = img_flat.mean(axis=1)
# 	elif way == 'weight':
# 		weights = np.array([0.2989, 0.5870, 0.1140])
# 		gray = img_flat @ weights
# 	else:
# 		raise ValueError("Invalid method. Use 'average' or 'weight'.")
#
# 	gray_3ch = np.repeat(gray[:, np.newaxis], 3, axis=1)
# 	return gray_3ch.reshape(row, col, 3)
#
# def change_grayscale_noflat(img_2d, way='weight'):
# 	if way == 'average':
# 		gray = img_2d.mean(axis=2, keepdims=True)
# 	elif way == 'weight':
# 		weights = np.array([0.2989, 0.5870, 0.1140])
# 		gray = (img_2d * weights).sum(axis=2, keepdims=True)
# 	else:
# 		raise ValueError("Invalid method. Use 'average' or 'weight'.")
#
# 	return np.repeat(gray, 3, axis=2)
#
# def c1(img_2d):
# 	weights = np.array([0.2989, 0.5870, 0.1140])
# 	gray = np.dot(img_2d, weights)[..., np.newaxis]
# 	return np.repeat(gray, 3, axis=2).astype(np.uint8)
# def c2(img_2d):
# 	weights = np.array([0.2989, 0.5870, 0.1140])
# 	# 'ijk,k->ij' means: sum over last axis (k), keeping i, j
# 	gray = np.einsum('ijk,k->ij', img_2d, weights)[..., np.newaxis]
# 	return np.repeat(gray, 3, axis=2).astype(np.uint8)
# def c3(img_2d):
# 	weights = np.array([0.2989, 0.5870, 0.1140])
# 	gray = np.tensordot(img_2d, weights, axes=([2], [0]))[..., np.newaxis]
# 	return np.repeat(gray, 3, axis=2).astype(np.uint8)
#
# def change_grayscale_neo(img_2d, way='weight'):
# 	width, height, channel = img_2d.shape
#
# 	# Flatten the image to (H*W, 3)
# 	img_flat = img_2d.reshape(-1, channel)
#
# 	# Weighted grayscale conversion using np.dot for efficiency
# 	if way == 'average':
# 		gray = img_flat.mean(axis=1)  # Average across channels
# 	elif way == 'weight':
# 		weights = np.array([0.2989, 0.5870, 0.1140])
# 		gray = np.dot(img_flat, weights)  # Dot product (RGB -> grayscale)
# 	else:
# 		raise ValueError("Invalid method. Use 'average' or 'weight'.")
#
# 	gray_3ch = np.repeat(gray[:, np.newaxis], 3, axis=1)
# 	return gray_3ch.reshape(width, height, 3).astype(np.uint8)

def rec601_luma_image(img):
	r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
	luma = 0.299 * r + 0.587 * g + 0.114 * b
	return np.repeat(luma[:, :, np.newaxis], 3, axis=-1).astype(np.uint8)

def rec709_luma_image(img):
	r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
	luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
	return np.repeat(luma[:, :, np.newaxis], 3, axis=-1).astype(np.uint8)

def rec601_luma_image2(img):
	r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
	luma = 0.299 * r + 0.587 * g + 0.114 * b
	return np.dstack([luma] * 3).astype(np.uint8) # broadcast the luma to all channels

def rec709_luma_image2(img):
	r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
	luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
	return np.dstack([luma] * 3).astype(np.uint8) # broadcast the luma to all channels

def rec601_luma_image3(img):
	r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
	luma = 0.299 * r + 0.587 * g + 0.114 * b
	return np.stack([luma] * 3, axis=-1).astype(np.uint8) # broadcast the luma to all channels

def rec709_luma_image3(img):
	img = img.copy()
	r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
	luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
	return np.stack([luma] * 3, axis=-1).astype(np.uint8) # broadcast the luma to all channels

def rec601_luma_image4(img):
	img = img.copy()
	r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
	luma = 0.2989 * r + 0.587 * g + 0.114 * b
	img[..., 0], img[..., 1], img[..., 2] = luma, luma, luma
	return img.astype(np.uint8) # broadcast the luma to all channels

def rec709_luma_image4(img):
	img = img.copy()
	r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
	luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
	img[..., 0], img[..., 1], img[..., 2] = luma, luma, luma
	return img.astype(np.uint8) # broadcast the luma to all channels

def rec601_luma_image5(img):
	img = img.copy()
	r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
	luma = 0.299 * r + 0.587 * g + 0.114 * b
	img[..., 0] = img[..., 1] = img[..., 2] = luma
	return img.astype(np.uint8) # broadcast the luma to all channels

def rec601_luma_image6(img):
	img = img.copy()
	img[..., 0] = img[..., 1] = img[..., 2] = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
	return img.astype(np.uint8)

def rec601_luma_image7(img):
	img = img.copy()
	luma = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
	img[..., 0], img[..., 1], img[..., 2] = luma, luma, luma
	return img.astype(np.uint8)

def rec601_luma_image8(img): # mod 2
	r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
	luma = 0.299 * r + 0.587 * g + 0.114 * b
	return np.dstack([luma, luma, luma]).astype(np.uint8) # broadcast the luma to all channels

def rec601_luma_image9(img): # mod 8
	r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
	luma = 0.299 * r + 0.587 * g + 0.114 * b
	if img.shape[2] == 4:
		return np.dstack([luma, luma, luma, img[:, :, 3]]).astype(np.uint8)
	return np.dstack([luma, luma, luma]).astype(np.uint8)

def rec601_luma_image10(img): # mod 9
	luma = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
	if img.shape[2] == 4:
		return np.dstack([luma, luma, luma, img[:, :, 3]]).astype(np.uint8)
	return np.dstack([luma, luma, luma]).astype(np.uint8)

def rec601_luma_image11(img): # mod 10
	img = img.copy()
	luma = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
	if img.shape[2] == 4:
		img[:, :, 0] = img[:, :, 1] = img[:, :, 2] = luma
		return img.astype(np.uint8)
	else:
		img[:, :, 0] = img[:, :, 1] = img[:, :, 2] = luma
		return img.astype(np.uint8)

def rec601_luma_image12(img): # winner
	luma = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
	result = np.zeros_like(img, dtype=np.uint8)  # initialize an empty result array
	result[:, :, 0] = result[:, :, 1] = result[:, :, 2] = luma
	if img.shape[2] == 4:
		result[:, :, 3] = img[:, :, 3]  # copy the alpha channel unchanged
	return result

def raw(img):
	return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

def raw_with_at(img):
	return (img[..., :3] @ np.array([0.299, 0.587, 0.114]).T).astype(np.uint8)

def raw_einsum(img):
	luma = np.einsum('ijk,k->ij', img[..., :3], [0.299, 0.587, 0.114])
	return np.dstack([luma] * 3).astype(np.uint8)

def raw_einsum3(img):
	# perform the luma calculation using Rec. 601 coefficients and update the 3rd dim
	img[..., 0], img[..., 1], img[..., 2] = [np.einsum('ijk,k->ij', img[..., :3], [0.299, 0.587, 0.114])] * 3
	return img.astype(np.uint8)

def raw_einsum4(img):
	img = img.copy()
	luma = np.einsum('ijk,k->ij', img[..., :3], [0.299, 0.587, 0.114])
	img[..., 0], img[..., 1], img[..., 2] = luma, luma, luma
	return img.astype(np.uint8)

from numba import jit
@jit(nopython=True)
def rec601_luma_image_numba(img):
	img = img.copy()
	# Assuming img is a NumPy array
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			r, g, b = img[i, j, 0], img[i, j, 1], img[i, j, 2]
			luma = 0.299 * r + 0.587 * g + 0.114 * b
			img[i, j, 0] = img[i, j, 1] = img[i, j, 2] = luma
	return img.astype(np.uint8)

img = read_img('../cat.jpg')
def run_and_measure_shuffled(funcs, img, runs=50):
	results = {name: {"times": [], "peak_mem": 0} for name, _ in funcs}

	for name, fn in funcs:
		tmp = fn(img)
		print(tmp.shape)
		print(f"Save to 'gray/{name}.png")
		save_img(tmp, "gray/" + name + '.png')
	save_img(img.astype(np.uint8), "./ori.png")
	for _ in range(runs):
		random.shuffle(funcs)  # 🌀 Shuffle run order each round
		for name, fn in funcs:
			tracemalloc.start()
			start_time = perf_counter()
			_ = fn(img)
			elapsed = perf_counter() - start_time
			current, peak = tracemalloc.get_traced_memory()
			tracemalloc.stop()

			results[name]["times"].append(elapsed)
			results[name]["peak_mem"] = max(results[name]["peak_mem"], peak)

	# Final aggregation
	for name in results:
		results[name]["avg_time"] = sum(results[name]["times"]) / len(results[name]["times"])
		results[name]["peak_mem_mib"] = results[name]["peak_mem"] / (1024 * 1024)

	return results
funcs = [
	# ('flat_weight',     lambda img: change_grayscale_flat(img)),
	# ('flat_avg',        lambda img: change_grayscale_flat(img, 'average')),
	# ('non_flat_avg',    lambda img: change_grayscale_noflat(img, 'average')),
	# ('non_flat_c2',     lambda img: c2(img)),
	# ('non_flat_c3',     lambda img: c3(img)),
	# ('neo_flat_weight', lambda img: change_grayscale_neo(img)),
	# ('neo_flat_avg',    lambda img: change_grayscale_neo(img, 'average')),
	# ('rec601',          lambda img: rec601_luma_image(img)),
	# ('rec709',          lambda img: rec709_luma_image(img)),
	('rec601_2',          lambda img: rec601_luma_image2(img)),
	# ('rec709_2',          lambda img: rec709_luma_image2(img)),
	('rec601_3',          lambda img: rec601_luma_image3(img)),
	# ('rec709_3',          lambda img: rec709_luma_image3(img)),
	# ('rec601_4',          lambda img: rec601_luma_image4(img)),
	# ('rec709_4',          lambda img: rec709_luma_image4(img)),
	# ('rec601_5',          lambda img: rec601_luma_image5(img)),
	# ('rec601_6',          lambda img: rec601_luma_image6(img)),
	('rec601_7',          lambda img: rec601_luma_image7(img)),
	('rec601_8',          lambda img: rec601_luma_image8(img)),
	('rec601_9',          lambda img: rec601_luma_image9(img)),
	('rec601_10',          lambda img: rec601_luma_image10(img)),
	# ('rec601_11',          lambda img: rec601_luma_image11(img)),
	('rec601_12',          lambda img: rec601_luma_image12(img)),
	# ('rec601_numba',      lambda img: rec601_luma_image_numba(img)),
	# ('raw',             lambda img: raw(img)),
	# ('raw_with_at',          lambda img: raw(img)),
	# ('raw_einsum',          lambda img: raw_einsum(img)),
	# ('raw_einsum3', lambda img: raw_einsum3(img)),
	# ('raw_einsum4', lambda img: raw_einsum4(img)),
]


# Method               | Avg Time (s) |  Peak Mem (MiB)
# --------------------------------------------------
# rec601_2             |     0.012159 |          11.597
# rec709_2             |     0.012595 |          11.597
# rec601_3             |     0.012230 |          11.597
# rec709_3             |     0.012494 |          11.597
# rec601_4             |     0.010397 |           7.325
# rec709_4             |     0.010407 |           7.325
# rec601_5             |     0.009958 |           7.325
# rec601_6             |     0.020032 |           7.325
# rec601_numba         |     0.004293 |           1.831
# raw_einsum           |     0.022905 |          21.363
# raw_einsum3          |     0.016753 |           5.072
# raw_einsum4          |     0.017045 |           6.714

# Method               | Avg Time (s) |  Peak Mem (MiB)
# --------------------------------------------------
# rec601_2             |     0.011618 |          11.597
# rec601_3             |     0.012091 |          11.597
# rec601_7             |     0.012449 |          14.649
# rec601_8             |     0.011753 |          11.597
# rec601_9             |     0.011440 |          11.597
# rec601_10            |     0.011653 |          11.597
# rec601_12            |     0.008399 |           7.325

# --- Run the benchmark ---
results = run_and_measure_shuffled(funcs, img, runs=200)
# --- Display results ---
print(f"{'Method':<20} | {'Avg Time (s)':>12} | {'Peak Mem (MiB)':>15}")
print("-" * 50)
for name, stats in results.items():
	print(f"{name:<20} | {stats['avg_time']:12.6f} | {stats['peak_mem_mib']:15.3f}")