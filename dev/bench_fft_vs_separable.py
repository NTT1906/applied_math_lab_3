import numpy as np
from PIL import Image
import matplotlib.pyplot as plt # for show image

def read_img(img_path):
	""" Read image from img_path
	returns a 2D image (numpy array)
	"""
	return np.array(Image.open(img_path))

def save_img(img_2d, img_path: str):
	"""	Save image to img_path
	"""
	Image.fromarray(img_2d).save(img_path)

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
	padded = np.pad(norm_img, ((pad, pad), (pad, pad), (0, 0)), mode='constant')

	# FFT size (minimal full convolution)
	fft_h = padded.shape[0] + k - 1
	fft_w = padded.shape[1] + k - 1

	# FFT of kernel using rfft2
	kernel_fft = np.fft.rfft2(kernel.astype(np.float32), s=(fft_h, fft_w))

	# Output size
	out_h = padded.shape[0] - k + 1
	out_w = padded.shape[1] - k + 1
	result = np.empty((out_h, out_w, c), dtype=np.float32)

	# Cropping indices
	r0, r1 = k - 1, k - 1 + out_h
	c0, c1 = k - 1, k - 1 + out_w

	# Process each channel
	for i in range(c):
		channel_fft = np.fft.rfft2(padded[:, :, i], s=(fft_h, fft_w))
		conv_fft = channel_fft * kernel_fft
		conv_real = np.fft.irfft2(conv_fft, s=(fft_h, fft_w))
		result[:, :, i] = conv_real[r0:r1, c0:c1]

	# Rescale and convert
	return np.clip(result * 255.0, 0, 255).astype(np.uint8)

def kernel_convolution_rfft_v2(img_2d: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	if kernel.ndim != 2:
		raise ValueError('Input kernel must be 2D')
	if img_2d.ndim != 3:
		raise ValueError('Input image must be 3D')

	# Normalize the image to [0, 1]
	norm_img = img_2d.astype(np.float32) / 255.0
	h, w, c = norm_img.shape
	if kernel.shape[0] != kernel.shape[1]:
		raise ValueError('Kernel must be square')

	k = kernel.shape[0]
	pad = k // 2
	padded_img = np.pad(norm_img, ((pad, pad), (pad, pad), (0, 0)), mode='constant')

	# fft size
	fft_h = padded_img.shape[0] + k - 1 # h + 2 * pad + k - 1 = 2 + (k//2) * 2 + k - 1
	fft_w = padded_img.shape[1] + k - 1

	# Compute the FFT of the kernel once
	kernel_fft = np.fft.rfft2(kernel.astype(np.float32), s=(fft_h, fft_w))

	# Prepare the result array (output size)
	out_h, out_w = h, w
	result = np.empty((out_h, out_w, c), dtype=np.float32)

	# Process each channel
	for i in range(c):
		# FFT of the current channel
		channel_fft = np.fft.rfft2(padded_img[:, :, i], s=(fft_h, fft_w))

		# Perform the convolution in the frequency domain
		conv_fft = channel_fft * kernel_fft

		# Inverse FFT to get the convolution result
		conv_real = np.fft.irfft2(conv_fft, s=(fft_h, fft_w))

		# Crop the result to the original image size
		result[:, :, i] = conv_real[pad:pad + out_h, pad:pad + out_w]

	# Rescale and convert back to uint8
	return np.clip(result * 255.0, 0, 255).astype(np.uint8)


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
		channel = img[:, :, c]

		# Vertical pass
		temp = np.apply_along_axis(lambda x: np.convolve(x, kernel_1d, mode='same'), axis=0, arr=channel)

		# Horizontal pass
		blurred = np.apply_along_axis(lambda x: np.convolve(x, kernel_1d, mode='same'), axis=1, arr=temp)

		output[:, :, c] = blurred

	return np.clip(output * 255.0, 0, 255).astype(np.uint8)

# def kernel_convolution_improved(img: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
#     if img.ndim != 3:
#         raise ValueError("Image must be 3D (H, W, C)")
#     if kernel_1d.ndim != 1:
#         raise ValueError("Kernel must be 1D")
#
#     img = img.astype(np.float32) / 255.0
#     H, W, C = img.shape
#     K = len(kernel_1d)
#     pad = K // 2
#
#     output = np.empty_like(img)
#
#     # Pad image for vertical pass
#     padded = np.pad(img, ((pad, pad), (0, 0), (0, 0)), mode='reflect')
#
#     # Vertical convolution (axis=0)
#     temp = np.zeros_like(img)
#     for i in range(K):
#         temp += kernel_1d[i] * padded[i:i+H, :, :]
#
#     # Pad temp for horizontal pass
#     temp_padded = np.pad(temp, ((0, 0), (pad, pad), (0, 0)), mode='reflect')
#
#     # Horizontal convolution (axis=1)
#     for i in range(K):
#         output += kernel_1d[i] * temp_padded[:, i:i+W, :]
#
#     return np.clip(output * 255.0, 0, 255).astype(np.uint8)

def kernel_convolution_apply2(img: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
	if img.ndim != 3:
		raise ValueError("Image must be 3D (H, W, C)")
	if kernel_1d.ndim != 1:
		raise ValueError("Kernel must be 1D")

	# Convert image to float32 and ensure contiguous memory for speed
	# img = np.ascontiguousarray(img.astype(np.float32)) / 255.0
	img = img.astype(np.float32) / 255.0
	H, W, C = img.shape

	output = np.empty_like(img)

	# Predefine correlation function (faster than convolve for symmetric kernels)
	def filter1d(x):
		return np.correlate(x, kernel_1d, mode='same')

	# Reuse temporary arrays to reduce allocations
	temp = np.empty((H, W), dtype=np.float32)
	blurred = np.empty((H, W), dtype=np.float32)

	for c in range(C):
		channel = img[:, :, c]
		# Vertical pass
		temp[:, :] = np.apply_along_axis(filter1d, axis=0, arr=channel)
		# Horizontal pass
		blurred[:, :] = np.apply_along_axis(filter1d, axis=1, arr=temp)
		output[:, :, c] = blurred

	# Rescale and convert to uint8
	return np.clip(output * 255.0, 0, 255).astype(np.uint8)

# def convolve_1d_strided(arr, kernel, axis):
# 	"""
# 	Perform 1D convolution along given axis using stride tricks and tensordot.
# 	Args:
# 		arr: 2D array (H, W)
# 		kernel: 1D kernel
# 		axis: 0 for vertical, 1 for horizontal convolution
# 	Returns:
# 		convolved 2D array of same shape as arr
# 	"""
# 	k = len(kernel)
# 	pad = k // 2
# 	if axis == 0:  # vertical
# 		padded = np.pad(arr, ((pad, pad), (0, 0)), mode='reflect')
# 		shape = (arr.shape[0], arr.shape[1], k)
# 		strides = (padded.strides[0], padded.strides[1], padded.strides[0])
# 		windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
# 		# tensordot along last dim (kernel)
# 		return np.tensordot(windows, kernel[::-1], axes=([2], [0]))
# 	elif axis == 1:  # horizontal
# 		padded = np.pad(arr, ((0, 0), (pad, pad)), mode='reflect')
# 		shape = (arr.shape[0], arr.shape[1], k)
# 		strides = (padded.strides[0], padded.strides[1], padded.strides[1])
# 		windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
# 		return np.tensordot(windows, kernel[::-1], axes=([2], [0]))
# 	else:
# 		raise ValueError("Axis must be 0 or 1")

# def kernel_convolution_separable_vec(img, k_row, k_col):
# 	"""
# 	Vectorized separable convolution for 3D image.
# 	Args:
# 		img: (H, W, C) uint8 image
# 		k_row, k_col: 1D kernels (horizontal and vertical)
# 	Returns:
# 		convolved uint8 image (H, W, C)
# 	"""
# 	img = img.astype(np.float32) / 255.0
# 	H, W, C = img.shape
# 	output = np.empty_like(img)
#
# 	for c in range(C):
# 		channel = img[:, :, c]
# 		# vertical conv
# 		v = convolve_1d_strided(channel, k_col, axis=0)
# 		# horizontal conv
# 		h = convolve_1d_strided(v, k_row, axis=1)
# 		output[:, :, c] = h
#
# 	return np.clip(output * 255, 0, 255).astype(np.uint8)

def kernel_convolution_direct(img: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
	if img.ndim != 3:
		raise ValueError("Image must be 3D (H, W, C)")
	if kernel_1d.ndim != 1:
		raise ValueError("Kernel must be 1D")

	img = img.astype(np.float32) / 255.0
	H, W, C = img.shape
	K = len(kernel_1d)
	pad = K // 2

	output = np.empty_like(img)

	# # Pad image vertically for vertical convolution
	# padded_v = np.pad(img, ((pad, pad), (0, 0), (0, 0)), mode='reflect')
	#
	# # Vertical convolution: sum weighted slices
	# temp = np.zeros((H, W, C), dtype=np.float32)
	# for i in range(K):
	#     temp += kernel_1d[i] * padded_v[i:i+H, :, :]
	#
	# # Pad horizontally for horizontal convolution
	# padded_h = np.pad(temp, ((0, 0), (pad, pad), (0, 0)), mode='reflect')
	#
	# # Horizontal convolution: sum weighted slices
	# for j in range(K):
	#     output += kernel_1d[j] * padded_h[:, j:j+W, :]
	# padded_v = np.pad(img, ((pad, pad), (0, 0), (0, 0)), mode='constant')
	# windows_v = np.stack([padded_v[i:i + H, :, :] for i in range(K)], axis=0)
	# temp = np.einsum('k,khwc->hwc', kernel_1d, windows_v)
	#
	# # Horizontal pass
	# padded_h = np.pad(temp, ((0, 0), (pad, pad), (0, 0)), mode='constant')
	# windows_h = np.stack([padded_h[:, j:j + W, :] for j in range(K)], axis=0)
	# output = np.einsum('k,khwc->hwc', kernel_1d, windows_h)

	for c in range(C):
		padded_v = np.pad(img[:, :, c], ((pad, pad), (0, 0)), mode='constant')
		windows_v = np.stack([padded_v[i:i + H, :] for i in range(K)], axis=0)  # (K, H, W)
		# einsum: sum over k, multiply kernel_1d[k] * windows_v[k, h, w]
		# temp = np.sum(kernel_1d[:, None, None] * windows_v, axis=0)
		temp = kernel_1d @ windows_v
		temp = temp.reshape(h, w)
		# temp = np.tensordot(kernel_1d, windows_v, axes=([0], [0]))
		# temp = np.einsum('k,khw->hw', kernel_1d, windows_v)

		padded_h = np.pad(temp, ((0, 0), (pad, pad)), mode='constant')
		windows_h = np.stack([padded_h[:, j:j + W] for j in range(K)], axis=0)  # (K, H, W)
		output[:, :, c] = np.einsum('k,khw->hw', kernel_1d, windows_h)

	return np.clip(output * 255.0, 0, 255).astype(np.uint8)

import timeit
img = read_img("../cat.jpg")

# num_trials = 100
num_trials = 200
v1 = []
v2 = []
v3 = []
v4 = []

k = np.array([1, 4, 6, 4, 1]) / 16
kernel_2d = np.outer(k, k)
assert is_separable_kernel(kernel_2d)
k_row, k_col = get_separable_components(kernel_2d)
print(k_row, k_col)

import tracemalloc
out = kernel_convolution_separable(img, k_row, k_col)
save_img(out, 'out_separable.png')
out = kernel_convolution_rfft(img, kernel_2d)
save_img(out, 'out_rfft.png')
tracemalloc.start()
out = kernel_convolution_apply(img, k_row)
current, peak = tracemalloc.get_traced_memory()
print(current, peak / (1024 * 1024))
tracemalloc.stop()
save_img(out, 'out_apply.png')
tracemalloc.start()
out = kernel_convolution_apply2(img, k_row)
current, peak = tracemalloc.get_traced_memory()
print(current, peak / (1024 * 1024))
tracemalloc.stop()
save_img(out, 'out_apply2.png')
tracemalloc.start()
# out = kernel_convolution_direct(img, k_row)
current, peak = tracemalloc.get_traced_memory()
print(current, peak / (1024 * 1024))
tracemalloc.stop()
save_img(out, 'out_separable_direct.png')

def test_v1():
	kernel_convolution_rfft(img, kernel_2d)

def test_v2():
	kernel_convolution_separable(img, k_row, k_col)

def test_v3():
	kernel_convolution_apply(img, k_row)

def test_v4():
	kernel_convolution_direct(img, k_row)

for _ in range(num_trials):
	# t1 = timeit.timeit(test_v1, number=1)
	# t4 = timeit.timeit(test_v4, number=1)
	t3 = timeit.timeit(test_v3, number=1)
	t2 = timeit.timeit(test_v2, number=1)
	# v1.append(t1)
	v2.append(t2)
	v3.append(t3)
	# v4.append(t4)

# Summary statistics
# print(f"Average time - v1: {np.mean(v1):.8f} sec")
print(f"Average time - v2: {np.mean(v2):.8f} sec")
print(f"Average time - v3: {np.mean(v3):.8f} sec")
# print(f"Average time - v4: {np.mean(v4):.8f} sec")

# Plotting
plt.figure(figsize=(10, 6))
# plt.plot(v1, label='v1')
plt.plot(v2, label='v2')
plt.plot(v3, label='v3')
# plt.plot(v4, label='v4')
plt.xlabel('Trial')
plt.ylabel('Time (seconds)')
plt.title('Performance Comparison: FFT')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
