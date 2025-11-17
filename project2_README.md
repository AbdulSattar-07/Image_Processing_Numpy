# Project 2: Image Processing Toolkit - Step-by-Step Guide

## 🖼️ Overview
This project teaches advanced NumPy concepts through practical image processing. You'll learn to manipulate images as multi-dimensional arrays, apply mathematical transformations, and create various visual effects.

## 🎯 Learning Objectives
- Master 3D array operations (height, width, RGB channels)
- Understand array broadcasting for image operations
- Learn convolution and filtering techniques
- Practice mathematical operations on images
- Implement color space transformations
- Create custom image filters

## 📁 Project Structure
```
project2_image_processing/
├── project2_image_processing.py      # Main implementation
├── project2_formulas_details.md      # Mathematical formulas
├── project2_README.md               # This guide
├── sample_images/                   # Test images folder
├── processed_images/                # Output folder
└── requirements.txt                 # Dependencies
```

## 🔧 Prerequisites
```bash
pip install numpy pillow matplotlib scipy
```

## 📊 Key NumPy Functions You'll Learn

| Function | Purpose | Example |
|----------|---------|---------|
| `np.array()` | Convert image to array | `np.array(PIL.Image.open('img.jpg'))` |
| `np.dot()` | Matrix multiplication | `np.dot(image, weights)` for grayscale |
| `np.clip()` | Clamp values to range | `np.clip(image, 0, 255)` |
| `np.where()` | Conditional selection | `np.where(mask, value1, value2)` |
| `np.convolve()` | Apply filters | `np.convolve(image, kernel)` |
| `np.histogram()` | Analyze distribution | `np.histogram(image, bins=256)` |
| `np.transpose()` | Rearrange axes | `np.transpose(image, (2, 0, 1))` |
| `np.concatenate()` | Join arrays | `np.concatenate([img1, img2], axis=1)` |

## 🚀 Step-by-Step Implementation

### Step 1: Image Loading and Basic Properties
```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load image
image = np.array(Image.open('sample.jpg'))
print(f"Image shape: {image.shape}")
print(f"Data type: {image.dtype}")
print(f"Value range: {image.min()} to {image.max()}")
```

**NumPy Concepts:**
- 3D array structure: `(height, width, channels)`
- Array properties: `shape`, `dtype`, `min()`, `max()`

### Step 2: Grayscale Conversion

**NumPy Concepts:**
- `np.dot()` for matrix multiplication
- Array slicing: `image[...,:3]`
- `axis` parameter in operations

**NumPy Concepts:**
- Channel indexing: `image[:, :, channel]`
- `np.zeros_like()` for creating similar arrays
- Selective assignment

axes[0,2].imshow(bright); axes[0,2].set_title('Brightened')
axes[1,0].imshow(contrast); axes[1,0].set_title('High Contrast')
axes[1,1].imshow(blurred); axes[1,1].set_title('Blurred')
axes[1,2].imshow(edges, cmap='gray'); axes[1,2].set_title('Edges')
plt.tight_layout()
plt.show()
```


This comprehensive guide provides everything needed to master image processing with NumPy!
