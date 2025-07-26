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
# ======== FUNCS =============
def kernel_convolution_rfft(img_2d: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	if kernel.ndim != 2:
		raise ValueError('Input kernel must be 2D')
	if img_2d.ndim != 3:
		raise ValueError('Input image must be 3D')
	norm_img = img_2d.astype(np.float32) / 255.0
	h, w, c = norm_img.shape
	if kernel.shape[0] != kernel.shape[1]:
		raise ValueError('Kernel must be square')

	k = kernel.shape[0]
	pad = k // 2
	padded_img = np.pad(norm_img, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
	fft_h = padded_img.shape[0] + k - 1
	fft_w = padded_img.shape[1] + k - 1
	kernel_fft = np.fft.rfft2(kernel.astype(np.float32), s=(fft_h, fft_w))
	result = np.empty((h, w, c), dtype=np.float32)
	r0, r1 = k - 1, padded_img.shape[0]
	c0, c1 = k - 1, padded_img.shape[1]
	for i in range(c):
		channel_fft = np.fft.rfft2(padded_img[:, :, i], s=(fft_h, fft_w))
		conv_fft = channel_fft * kernel_fft
		conv_real = np.fft.irfft2(conv_fft, s=(fft_h, fft_w))
		result[:, :, i] = conv_real[r0:r1, c0:c1]
	return np.clip(result * 255.0, 0, 255).astype(np.uint8)

def kernel_convolution_rfft_smallfft(img_2d: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	if kernel.ndim != 2:
		raise ValueError('Input kernel must be 2D')
	if img_2d.ndim != 3:
		raise ValueError('Input image must be 3D')
	if kernel.shape[0] != kernel.shape[1]:
		raise ValueError('Kernel must be square')

	norm_img = img_2d.astype(np.float32) / 255.0
	k = kernel.shape[0]
	pad = k // 2
	padded_img = np.pad(norm_img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect') # constant -> 0, reflect -> copy the edge -> no black edge

	# Smaller FFT size
	fft_h = padded_img.shape[0]
	fft_w = padded_img.shape[1]

	kernel_fft = np.fft.rfft2(kernel.astype(np.float32), s=(fft_h, fft_w))
	channel_fft = np.fft.rfft2(padded_img, s=(fft_h, fft_w), axes=(0, 1))
	conv_fft = channel_fft * kernel_fft[:, :, np.newaxis]
	conv_real = np.fft.irfft2(conv_fft, s=(fft_h, fft_w), axes=(0, 1))
	conv_real *= (fft_h * fft_w) / (img_2d.shape[0] * img_2d.shape[1])

	# Adjusted cropping for smaller FFT
	# r0, r1 = k // 2 + 2, k // 2 + h + 2
	# c0, c1 = k // 2 + 2, k // 2 + w + 2
	r0, r1 = k - 1, padded_img.shape[0]
	c0, c1 = k - 1, padded_img.shape[1]
	result = conv_real[r0:r1, c0:c1, :]

	return np.clip(result * 255.0, 0, 255).astype(np.uint8)

def kernel_convolution_rfft_uint8(img_2d: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	if kernel.ndim != 2:
		raise ValueError('Input kernel must be 2D')
	if img_2d.ndim != 3:
		raise ValueError('Input image must be 3D')
	norm_img = img_2d.astype(np.float32)  # Keep [0, 255]
	h, w, c = norm_img.shape
	if kernel.shape[0] != kernel.shape[1]:
		raise ValueError('Kernel must be square')

	# Normalize kernel to sum to 1
	kernel = kernel.astype(np.float32)
	kernel_sum = kernel.sum()
	if kernel_sum != 0:
		kernel = kernel / kernel_sum

	k = kernel.shape[0]
	pad = k // 2
	padded_img = np.pad(norm_img, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
	fft_h = padded_img.shape[0] + k - 1
	fft_w = padded_img.shape[1] + k - 1
	kernel_fft = np.fft.rfft2(kernel, s=(fft_h, fft_w))
	result = np.empty((h, w, c), dtype=np.float32)
	r0, r1 = k - 1, padded_img.shape[0]
	c0, c1 = k - 1, padded_img.shape[1]
	for i in range(c):
		channel_fft = np.fft.rfft2(padded_img[:, :, i], s=(fft_h, fft_w))
		conv_fft = channel_fft * kernel_fft
		conv_real = np.fft.irfft2(conv_fft, s=(fft_h, fft_w))
		result[:, :, i] = conv_real[r0:r1, c0:c1]
	return np.clip(result, 0, 255).astype(np.uint8)

# ======== UTILS =============
def kernel_identity(size: int) -> np.array:
	return np.pad([[1]], pad_width=size//2)

def kernel_box_blur(size: int) -> np.ndarray:
	return np.ones((size, size)) / size ** 2

def kernel_gauss_blur(size: int) -> np.ndarray:
	if size % 2 == 0 or size < 1:
		raise ValueError("Size must be odd and positive")
	kernel_1d = np.array([1, 1])
	for _ in range(size - 2):
		kernel_1d = np.convolve(kernel_1d, [1, 1])
	return np.outer(kernel_1d, kernel_1d) / (4 ** (size - 1))
# ======== BENCHMARK =============
img = read_img('../cat.jpg')
gauss5 = kernel_gauss_blur(5)

def run_and_measure_shuffled(funcs, img, runs=50):
	results = {name: {"times": [], "peak_mem": 0} for name, _ in funcs}

	for name, fn in funcs:
		tracemalloc.start()
		tmp = fn(img)
		current, peak = tracemalloc.get_traced_memory()
		tracemalloc.stop()
		results[name]["peak_mem"] = peak
		print(tmp.shape, peak)
		print(f"Save to 'convolution/{name}.png")
		save_img(tmp, "convolution/" + name + '.png')
	save_img(img.astype(np.uint8), "./ori.png")
	for _ in range(runs):
		random.shuffle(funcs)  # shuffle run order each round to avoid bias
		for name, fn in funcs:
			start_time = perf_counter()
			_ = fn(img)
			elapsed = perf_counter() - start_time

			results[name]["times"].append(elapsed)

	# Final aggregation
	for name in results:
		results[name]["avg_time"] = sum(results[name]["times"]) / len(results[name]["times"])
		results[name]["peak_mem_mib"] = results[name]["peak_mem"] / (1024 * 1024)

	return results

funcs = [
	('rfft', lambda img: kernel_convolution_rfft(img, gauss5)),
	('rfft_smallfft', lambda img: kernel_convolution_rfft_smallfft(img, gauss5)),
	('rfft_uint8', lambda img: kernel_convolution_rfft_uint8(img, gauss5)),
]
# import cProfile
# cProfile.run("kernel_convolution_rfft(img, gauss5)")

# --- Run the benchmark ---
results = run_and_measure_shuffled(funcs, img, runs=50)
# --- Display results ---
print(f"{'Method':<20} | {'Avg Time (s)':>12} | {'Peak Mem (MiB)':>15}")
print("-" * 50)
for name, stats in results.items():
	print(f"{name:<20} | {stats['avg_time']:12.6f} | {stats['peak_mem_mib']:15.3f}")