import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.widgets import Slider

def read_img(img_path):
	""" Read image from img_path
	returns a 2D image (numpy array)
	"""
	return np.array(Image.open(img_path)) # (width, height, channel)

def simple_blur(image):
	kernel_size = 3
	pad = kernel_size // 2
	blurred = np.zeros_like(image, dtype=np.float32)

	# Process each channel independently
	for c in range(image.shape[2]):
		padded = np.pad(image[:, :, c], ((pad, pad), (pad, pad)), mode='edge')
		for i in range(image.shape[0]):
			for j in range(image.shape[1]):
				blurred[i, j, c] = np.mean(padded[i:i + kernel_size, j:j + kernel_size])
	return blurred


def blend_images(original, blurred, alpha):
	blended = (1 - alpha) * original + alpha * blurred
	return np.clip(blended, 0, 255).astype(np.uint8)


# Create a sample grayscale image (e.g., gradient + noise)
height, width = 100, 100
original_img = read_img('../cat.jpg')

blurred_img = simple_blur(original_img)

# Initial alpha
alpha_init = 0.5

# Plot setup
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

l = ax.imshow(blend_images(original_img, blurred_img, alpha_init), cmap='gray')
ax.set_title('Alpha Blending of Original and Blurred Image')
ax.axis('off')

# Slider axis
ax_alpha = plt.axes([0.25, 0.1, 0.50, 0.03])
slider_alpha = Slider(ax_alpha, 'Alpha', -15.0, 15.0, valinit=alpha_init)


# Update function
def update(val):
	alpha = slider_alpha.val
	blended = blend_images(original_img, blurred_img, alpha)
	l.set_data(blended)
	fig.canvas.draw_idle()


slider_alpha.on_changed(update)

plt.show()
