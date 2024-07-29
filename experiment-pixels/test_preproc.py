# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import scipy.misc

# # Preproc function with the provided code
# def preproc(X, side):
#     '''Preprocess an image for input into the model'''
#     from PIL import Image
#     img = Image.fromarray(X)
#     img = img.resize((int(side), side), Image.Resampling.LANCZOS)
#     return np.array(img).flatten() / 255.0  # Flatten the image to a 1D array

# # Load the image
# image_path = 'pendulum.png'
# image = Image.open(image_path)
# image_np = np.array(image)

# # Check the shape of the image to understand the dimensions
# print("Original image shape:", image_np.shape)

# # Apply the preproc function
# side = 28  # Example side length for resizing
# processed_image = preproc(image_np, side)

# # Check the shape of the processed image
# print("Processed image shape:", processed_image.shape)

# # Reshape the processed image back to 2D for visualization
# processed_image_2d = processed_image.reshape(side, side)

# # Display the original and processed images
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(image_np)
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title("Processed Image")
# plt.imshow(processed_image_2d, cmap='gray')
# plt.axis('off')

# plt.show()

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Preproc function with the provided code
def preproc(X, side):
    '''Preprocess an image for input into the model'''
    from PIL import Image
    img = Image.fromarray(X)
    img = img.resize((int(side), side), Image.Resampling.LANCZOS)
    return np.array(img).flatten() / 255.0  # Flatten the image to a 1D array

# Load the image
image_path = 'pendulum.png'
image = Image.open(image_path).convert('RGB')  # Convert to RGB to remove alpha channel
image_np = np.array(image)

# Check the shape of the image to understand the dimensions
print("Original image shape:", image_np.shape)

# Apply the preproc function
side = 28  # Example side length for resizing
processed_image = preproc(image_np, side)

# Check the shape of the processed image
print("Processed image shape:", processed_image.shape)

# Reshape the processed image back to 2D for visualization
processed_image_2d = processed_image.reshape(side, side, 3)  # Reshape for RGB image

# Display the original and processed images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_np)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Processed Image")
plt.imshow(processed_image_2d)
plt.axis('off')

plt.show()

