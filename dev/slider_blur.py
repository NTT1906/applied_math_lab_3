import numpy as np
import matplotlib.pyplot as plt
# from scipy.ndimage import convolve
from matplotlib.widgets import Slider

import matplotlib.pyplot as plt
from PIL import Image
import os

def read_img(filepath):
	img = Image.open(filepath).convert('RGB')
	return np.array(img, dtype=np.float32)

def save_img(img_arr, filepath):
	os.makedirs(os.path.dirname(filepath), exist_ok=True)
	Image.fromarray(img_arr).save(filepath)

def gauss_kernel(m, n, sigma):
	x = np.arange(-(m - 1) // 2, (m - 1) // 2 + 1)
	y = np.arange(-(n - 1) // 2, (n - 1) // 2 + 1)
	xx, yy = np.meshgrid(x, y)
	kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
	kernel = kernel / np.sum(kernel)
	return kernel

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

# Load a sample grayscale image
image = read_img("../cat.jpg")
# Parameters for the Gaussian kernel
kernel_size = 7  # Size of the kernel (odd number)
# Create the Gaussian kernel initially with a default sigma
sigma_init = 2.0
kernel = gauss_kernel(kernel_size, kernel_size, sigma_init)
blurred_image = kernel_convolution_rfft(image, kernel)

# Plot the original and blurred images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')

blurred_im_ax = axes[1].imshow(blurred_image)
axes[1].set_title("Blurred Image")
axes[1].axis('off')

# Create a slider for sigma
ax_sigma = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
sigma_slider = Slider(ax_sigma, 'Sigma', 0.1, 10.0, valinit=sigma_init, valstep=0.1)

def update(val):
	sigma = sigma_slider.val
	kernel = gauss_kernel(kernel_size, kernel_size, sigma)  # Create new kernel with updated sigma
	blurred_image = kernel_convolution_rfft(image, kernel)  # Apply convolution
	blurred_im_ax.set_data(blurred_image)  # Update the displayed blurred image
	fig.canvas.draw_idle()  # Redraw the canvas

# Register the update function with the slider
sigma_slider.on_changed(update)

# Display the plot
plt.show()
