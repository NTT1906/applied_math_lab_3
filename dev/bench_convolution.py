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
	padded_img = np.pad(norm_img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
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

def kernel_convolution_stride(img_2d: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	if kernel.ndim != 2:
		raise ValueError('Input kernel must be 2D')
	if img_2d.ndim != 3:
		raise ValueError('Input image must be 3D')

	norm_img = img_2d.astype(np.float16) / 255.0
	height, width, _ = norm_img.shape
	k_height, k_width = kernel.shape
	if k_height != k_width:
		raise ValueError('Input kernel matrix must be square')

	# pad image to handle boundaries
	pad_size = k_height // 2
	pad_img = np.pad(norm_img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')

	patches = np.lib.stride_tricks.sliding_window_view(pad_img, window_shape=(k_height, k_height), axis=(0, 1)) # shape: (height, width, channel, k_height, k_width)
	result = np.einsum('abcde,de->abc', patches, kernel)
	result = np.clip(result * 255, 0, 255).astype(np.uint8)
	return result

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
	padded_img = np.pad(norm_img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
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

def is_separable_kernel(kernel: np.ndarray, tol=1e-5) -> bool:
	if kernel.ndim != 2:
		raise ValueError("Kernel must be 2D")
	u, s, vh = np.linalg.svd(kernel, full_matrices=False)
	return np.sum(s > tol) == 1


def get_separable_components(kernel: np.ndarray, tol=1e-5):
	if not is_separable_kernel(kernel, tol):
		raise ValueError("Kernel is not separable")

	u, s, vh = np.linalg.svd(kernel, full_matrices=False)
	root_s = np.sqrt(s[0])
	col = u[:, 0] * root_s
	row = vh[0, :] * root_s
	return row, col  # row first (horizontal), then column (vertical)

def kernel_convolution_separable(img: np.ndarray, k_row: np.ndarray, k_col: np.ndarray) -> np.ndarray:
	if img.ndim != 3:
		raise ValueError("Image must be 3D (H, W, C)")
	if k_row.ndim != 1 or k_col.ndim != 1:
		raise ValueError("Kernels must be 1D")

	img = img.astype(np.float32) / 255.0
	H, W, C = img.shape
	output = np.empty_like(img)

	pad_r = len(k_col) // 2
	pad_c = len(k_row) // 2

	for c in range(C):
		channel = img[:, :, c]

		# Pad and convolve vertically
		padded_v = np.pad(channel, ((pad_r, pad_r), (0, 0)), mode='reflect')
		temp = np.zeros_like(channel)
		for i in range(len(k_col)):
			temp += k_col[i] * padded_v[i:i+H, :]

		# Pad and convolve horizontally
		padded_h = np.pad(temp, ((0, 0), (pad_c, pad_c)), mode='reflect')
		final = np.zeros_like(channel)
		for j in range(len(k_row)):
			final += k_row[j] * padded_h[:, j:j+W]

		output[:, :, c] = final

	return np.clip(output * 255, 0, 255).astype(np.uint8)

def kernel_convolution_apply(img: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
	if img.ndim != 3:
		raise ValueError("Image must be 3D (H, W, C)")
	if kernel_1d.ndim != 1:
		raise ValueError("Kernel must be 1D")

	img = img.astype(np.float32) / 255.0
	H, W, C = img.shape
	output = np.empty_like(img)

	for c in range(C):
		temp = np.apply_along_axis(lambda x: np.convolve(x, kernel_1d, mode='same'), axis=0, arr=img[:, :, c])
		final = np.apply_along_axis(lambda x: np.convolve(x, kernel_1d, mode='same'), axis=1, arr=temp)
		output[:, :, c] = final

	return np.clip(output * 255.0, 0, 255).astype(np.uint8)

def kernel_convolution_apply2(img: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
	if img.ndim != 3:
		raise ValueError("Image must be 3D (H, W, C)")
	if kernel_1d.ndim != 1:
		raise ValueError("Kernel must be 1D")

	img = img.astype(np.float32) / 255.0
	H, W, C = img.shape
	output = np.empty_like(img)

	tmp = np.zeros_like(img[:, :, 0])
	for c in range(C):
		tmp = np.apply_along_axis(lambda x: np.convolve(x, kernel_1d, mode='same'), axis=0, arr=img[:, :, c])
		output[:, :, c] = np.apply_along_axis(lambda x: np.convolve(x, kernel_1d, mode='same'), axis=1, arr=tmp)

	return np.clip(output * 255.0, 0, 255).astype(np.uint8)


def kernel_convolution_apply3(img: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
	if img.ndim != 3:
		raise ValueError("Image must be 3D (H, W, C)")
	if kernel_1d.ndim != 1:
		raise ValueError("Kernel must be 1D")

	img = img.astype(np.float32) / 255.0
	H, W, C = img.shape

	img_padded = np.pad(img, ((kernel_1d.size // 2, kernel_1d.size // 2),
							  (kernel_1d.size // 2, kernel_1d.size // 2),
							  (0, 0)), mode='reflect')
	# print('A3', img_padded.shape)


	tmp = np.zeros_like(img[:, :, 0])
	for c in range(C):
		# print('1: ', tmp.shape)
		tmp = np.apply_along_axis(lambda x: np.convolve(x, kernel_1d, mode='valid'), axis=0, arr=img_padded[:, :, c])
		# print('2: ', tmp.shape)
		img[:, :, c] = np.apply_along_axis(lambda x: np.convolve(x, kernel_1d, mode='valid'), axis=1, arr=tmp)
		# print('3: ', img[:, :, c].shape)

	return np.clip(img * 255.0, 0, 255).astype(np.uint8)

def kernel_convolution_apply4(img: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
	if img.ndim != 3:
		raise ValueError("Image must be 3D (H, W, C)")
	if kernel_1d.ndim != 1:
		raise ValueError("Kernel must be 1D")

	img = img.astype(np.float32) / 255.0
	H, W, C = img.shape

	img_padded = np.pad(img, ((kernel_1d.size // 2, kernel_1d.size // 2),
							  (kernel_1d.size // 2, kernel_1d.size // 2),
							  (0, 0)), mode='reflect')
	# print('A4', img_padded.shape)

	for c in range(C):
		img_padded[:H, :, c] = np.apply_along_axis(lambda x: np.convolve(x, kernel_1d, mode='valid'), axis=0, arr=img_padded[:, :, c])
		img[:, :, c] = np.apply_along_axis(lambda x: np.convolve(x, kernel_1d, mode='valid'), axis=1, arr=img_padded[:H, :, c])

	return np.clip(img * 255.0, 0, 255).astype(np.uint8)

def kernel_convolution_apply5(img: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
	if img.ndim != 3:
		raise ValueError("Image must be 3D (H, W, C)")
	if kernel_1d.ndim != 1:
		raise ValueError("Kernel must be 1D")
	H, W, C = img.shape

	img = np.pad(img.astype(np.float32) / 255.0, ((kernel_1d.size // 2, kernel_1d.size // 2),
							  (kernel_1d.size // 2, kernel_1d.size // 2),
							  (0, 0)), mode='reflect')

	for c in range(C):
		img[:H, :, c] = np.apply_along_axis(lambda x: np.convolve(x, kernel_1d, mode='valid'), axis=0, arr=img[:, :, c])
		img[:H, :W, c] = np.apply_along_axis(lambda x: np.convolve(x, kernel_1d, mode='valid'), axis=1, arr=img[:H, :, c])

	return np.clip(img[:H,:W,:] * 255.0, 0, 255).astype(np.uint8)


def kernel_convolution_apply6(img: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
	if img.ndim != 3:
		raise ValueError("Image must be 3D (H, W, C)")
	if kernel_1d.ndim != 1:
		raise ValueError("Kernel must be 1D")

	H, W, C = img.shape
	k_size = kernel_1d.size
	pad_size = k_size // 2
	pad_end_height, pad_end_width = pad_size + H, pad_size + W

	# Normalize and pad the image
	img = np.pad(img.astype(np.float32) / 255.0,((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')

	# Apply convolution directly on the padded image for each channel
	for c in range(C):
		img[pad_size:pad_end_height, :, c] = np.apply_along_axis(lambda x: np.convolve(x, kernel_1d, mode='valid'), axis=0, arr=img[:, :, c])
		img[pad_size:pad_end_height, pad_size:pad_end_width, c] = np.apply_along_axis(lambda x: np.convolve(x, kernel_1d, mode='valid'), axis=1, arr=img[pad_size:pad_end_height, :, c])

	# Clip and return the result
	return np.clip(img[pad_size:pad_end_height, pad_size:pad_end_width, :] * 255.0, 0, 255).astype(np.uint8)

def kernel_convolution_apply7(img: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
	if img.ndim != 3:
		raise ValueError("Image must be 3D (H, W, C)")
	if kernel_1d.ndim != 1:
		raise ValueError("Kernel must be 1D")

	H, W, C = img.shape
	k_size = kernel_1d.size
	pad_size = k_size // 2
	pad_end_height, pad_end_width = pad_size + H, pad_size + W

	# Normalize and pad the image
	img = np.pad(img.astype(np.float32) / 255.0,((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')

	# Apply convolution directly on the padded image for each channel
	for c in range(C):
		img[:, :, c] = np.apply_along_axis(lambda x: np.convolve(x, kernel_1d, mode='same'), axis=0, arr=img[:, :, c])
		img[:, :, c] = np.apply_along_axis(lambda x: np.convolve(x, kernel_1d, mode='same'), axis=1, arr=img[:, :, c])

	# Clip and return the result
	return np.clip(img[pad_size:pad_end_height, pad_size:pad_end_width, :] * 255.0, 0, 255).astype(np.uint8)

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
MAT_UNSHARP_BLUR_5 = -0.00390625 * np.array(
	[[1,  4,    6,  4,  1],
	 [4, 16,   24, 16, 14],
	 [6, 24, -476, 24,  6],
	 [4, 16,   24, 16,  4],
	 [1,  4,    6,  4,  1]])

img = read_img('../cat.jpg')
gauss5 = kernel_gauss_blur(31)
# gauss5 = MAT_UNSHARP_BLUR_5
k_row, k_col = get_separable_components(gauss5)
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
	# ('stride', lambda img: kernel_convolution_stride(img, gauss5)),
	('separate', lambda img: kernel_convolution_separable(img, k_row, k_col)),
	('covo_apply', lambda img: kernel_convolution_apply(img, k_row)),
	('covo_apply2', lambda img: kernel_convolution_apply2(img, k_row)),
	('covo_apply3', lambda img: kernel_convolution_apply3(img, k_row)),
	('covo_apply4', lambda img: kernel_convolution_apply4(img, k_row)),
	('covo_apply5', lambda img: kernel_convolution_apply5(img, k_row)),
	('covo_apply6', lambda img: kernel_convolution_apply6(img, k_row)),
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