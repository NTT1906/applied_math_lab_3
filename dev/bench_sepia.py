import random
import numpy as np
import timeit

def sepia_reshape_dot(img):
	img_1d = img.astype(np.float32).reshape(-1, 3)
	sepia_filter = np.array([
		[0.393, 0.769, 0.189],
		[0.349, 0.686, 0.168],
		[0.272, 0.534, 0.131]
	], dtype=np.float32)

	result = np.dot(img_1d, sepia_filter.T)
	result = np.clip(result, 0, 255)
	return result.reshape(img.shape).astype(np.uint8)

def sepia_reshape_dot_og(img):
	img_2d = img.astype(np.float32)
	sepia_filter = np.array([
		[0.393, 0.769, 0.189],
		[0.349, 0.686, 0.168],
		[0.272, 0.534, 0.131]
	], dtype=np.float32)

	img_1d = img_2d.reshape(-1, 3)

	res_1d = np.dot(img_1d, sepia_filter.T)
	res_1d = np.clip(res_1d, 0, 255)
	res_2d = res_1d.reshape(img.shape)
	return res_2d.astype(np.uint8)

def sepia_matmul(img):
	img_2d = img.astype(np.float32)
	sepia_filter = np.array([
		[0.393, 0.769, 0.189],
		[0.349, 0.686, 0.168],
		[0.272, 0.534, 0.131]
	], dtype=np.float32)

	res = img_2d @ sepia_filter.T
	res = np.clip(res, 0, 255)
	return res.astype(np.uint8)

def sepia_einsum(img):
	img = img.astype(np.float32)

	# Sepia filter matrix (3x3)
	sepia_filter = np.array([
		[0.393, 0.769, 0.189],
		[0.349, 0.686, 0.168],
		[0.272, 0.534, 0.131]
	], dtype=np.float32)

	# Use einsum to apply the filter: sum over the color channels
	# '...c,dc->...d' means:
	#   - img: (..., c) → HWC (c = channels)
	#   - sepia_filter: (d, c)
	#   - output: (..., d) → HW3
	sepia_img = np.einsum('...c,dc->...d', img, sepia_filter)

	return np.clip(sepia_img, 0, 255).astype(np.uint8)


# Create a dummy image (e.g., 1080p RGB)
img = np.random.randint(0, 256, size=(1080, 1920, 3), dtype=np.uint8)

# # Timing
# num_runs = 100
# time_reshape_dot = timeit.timeit(lambda: sepia_reshape_dot(img), number=num_runs)
# time_reshape_dot_og = timeit.timeit(lambda: sepia_reshape_dot_og(img), number=num_runs)
#
# print(f"Reshape + dot : {time_reshape_dot / num_runs:.8f} seconds per run")
# print(f"Reshape + dot2: {time_reshape_dot_og / num_runs:.8f} seconds per run")
# # print(f"Direct matmul (@): {np.sum(v2) / num_runs:.8f} seconds per run")
# # print(f"Eisum            : {np.sum(v3) / num_runs:.8f} seconds per run")

funcs = [
	("Reshape + dot", lambda: sepia_reshape_dot(img)),
	("Reshape + dot2", lambda: sepia_reshape_dot_og(img))
]

# Warm-up both
for name, func in funcs:
	func()

# Randomize order to avoid CPU cache bias
results = {name: [] for name, _ in funcs}
amount = 200
for _ in range(amount):
	random.shuffle(funcs)
	for name, func in funcs:
		t = timeit.timeit(func, number=1)
		results[name].append(t)

# Print average
for name in results:
	avg = sum(results[name]) / len(results[name])
	print(f"{name}: {avg:.8f} seconds per run")
