import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image

# --- Read and preprocess image ---
def read_img(img_path):
    """Read image from path as numpy array"""
    return np.array(Image.open(img_path))

# --- Generate unsharp mask kernel ---
def unsharp_kernel(n=5, lambda_=1.0):
    from scipy.special import comb
    if n % 2 == 0 or n < 3:
        raise ValueError("Kernel size must be odd and >= 3")

    # Gaussian kernel via Pascal's triangle
    row = np.array([comb(n - 1, i, exact=True) for i in range(n)])
    gauss = np.outer(row, row)
    gauss = gauss / np.sum(gauss)

    identity = np.zeros_like(gauss)
    center = n // 2
    identity[center, center] = 1

    return (1 + lambda_) * identity - lambda_ * gauss

# --- Convolution with rfft ---
def kernel_convolution_rfft(img_2d: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if kernel.ndim != 2:
        raise ValueError('Input kernel must be 2D')
    if img_2d.ndim != 3:
        raise ValueError('Input image must be 3D')

    norm_img = img_2d.astype(np.float32) / 255.0
    h, w, c = norm_img.shape
    k = kernel.shape[0]
    pad = k // 2
    padded = np.pad(norm_img, ((pad, pad), (pad, pad), (0, 0)), mode='constant')

    fft_h = padded.shape[0] + k - 1
    fft_w = padded.shape[1] + k - 1
    kernel_fft = np.fft.rfft2(kernel.astype(np.float32), s=(fft_h, fft_w))

    out_h = padded.shape[0] - k + 1
    out_w = padded.shape[1] - k + 1
    result = np.empty((out_h, out_w, c), dtype=np.float32)

    r0, r1 = k - 1, k - 1 + out_h
    c0, c1 = k - 1, k - 1 + out_w

    for i in range(c):
        channel_fft = np.fft.rfft2(padded[:, :, i], s=(fft_h, fft_w))
        conv_fft = channel_fft * kernel_fft
        conv_real = np.fft.irfft2(conv_fft, s=(fft_h, fft_w))
        result[:, :, i] = conv_real[r0:r1, c0:c1]

    return np.clip(result * 255.0, 0, 255).astype(np.uint8)

# --- Load image ---
img = read_img("../cat.jpg")

# --- Set up plot ---
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Initial lambda
lambda_init = 1.0
processed = kernel_convolution_rfft(img, unsharp_kernel(n=5, lambda_=lambda_init))
img_plot = ax.imshow(processed)
ax.set_title(f"Unsharp Mask (λ = {lambda_init})")
ax.axis("off")

# --- Add slider ---
ax_lambda = plt.axes([0.25, 0.1, 0.5, 0.03])
slider = Slider(ax_lambda, label='λ (sharpen)', valmin=-1.0, valmax=3.0, valinit=lambda_init, valstep=0.1)

# --- Slider update ---
def update(val):
    lam = slider.val
    kernel = unsharp_kernel(n=5, lambda_=lam)
    sharpened = kernel_convolution_rfft(img, kernel)
    img_plot.set_data(sharpened)
    ax.set_title(f"Unsharp Mask (λ = {lam:.1f})")
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
