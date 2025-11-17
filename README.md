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
```python
def rgb_to_grayscale(image):
    """Convert RGB image to grayscale using luminance weights"""
    # Method 1: Luminance weights (recommended)
    grayscale = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    
    # Method 2: Simple average
    # grayscale = np.mean(image, axis=2)
    
    return grayscale.astype(np.uint8)
```

**NumPy Concepts:**
- `np.dot()` for matrix multiplication
- Array slicing: `image[...,:3]`
- `axis` parameter in operations

### Step 3: Brightness and Contrast Adjustment
```python
def adjust_brightness(image, brightness):
    """Adjust image brightness"""
    return np.clip(image.astype(np.float32) + brightness, 0, 255).astype(np.uint8)

def adjust_contrast(image, contrast):
    """Adjust image contrast"""
    # Center around 128, apply contrast, restore center
    adjusted = (image.astype(np.float32) - 128) * contrast + 128
    return np.clip(adjusted, 0, 255).astype(np.uint8)
```

**NumPy Concepts:**
- Broadcasting: scalar operations on arrays
- `np.clip()` for value clamping
- Data type conversion: `astype()`

### Step 4: Color Channel Manipulation
```python
def split_channels(image):
    """Split image into color channels"""
    red = image[:, :, 0]
    green = image[:, :, 1] 
    blue = image[:, :, 2]
    return red, green, blue

def create_color_filter(image, channel='red'):
    """Create single-color version of image"""
    filtered = np.zeros_like(image)
    
    if channel == 'red':
        filtered[:, :, 0] = image[:, :, 0]
    elif channel == 'green':
        filtered[:, :, 1] = image[:, :, 1]
    elif channel == 'blue':
        filtered[:, :, 2] = image[:, :, 2]
    
    return filtered
```

**NumPy Concepts:**
- Channel indexing: `image[:, :, channel]`
- `np.zeros_like()` for creating similar arrays
- Selective assignment

### Step 5: Image Filtering with Convolution
```python
def create_blur_kernel(size=3):
    """Create Gaussian blur kernel"""
    kernel = np.ones((size, size), dtype=np.float32)
    kernel = kernel / kernel.sum()  # Normalize
    return kernel

def create_sharpen_kernel():
    """Create sharpening kernel"""
    kernel = np.array([[ 0, -1,  0],
                      [-1,  5, -1],
                      [ 0, -1,  0]], dtype=np.float32)
    return kernel

def apply_filter(image, kernel):
    """Apply convolution filter to image"""
    if len(image.shape) == 3:
        # Apply to each channel separately
        filtered = np.zeros_like(image)
        for i in range(image.shape[2]):
            filtered[:, :, i] = convolve2d(image[:, :, i], kernel)
        return filtered
    else:
        # Grayscale image
        return convolve2d(image, kernel)
```

**NumPy Concepts:**
- Kernel creation with `np.array()`
- Normalization: `kernel / kernel.sum()`
- Loop over channels

### Step 6: Edge Detection
```python
def sobel_edge_detection(image):
    """Apply Sobel edge detection"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = rgb_to_grayscale(image)
    else:
        gray = image
    
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float32)
    
    # Apply filters
    edges_x = convolve2d(gray, sobel_x)
    edges_y = convolve2d(gray, sobel_y)
    
    # Calculate magnitude
    edges = np.sqrt(edges_x**2 + edges_y**2)
    
    return np.clip(edges, 0, 255).astype(np.uint8)
```

**NumPy Concepts:**
- `np.sqrt()` for mathematical operations
- Element-wise operations: `**2`
- Combined array operations

### Step 7: Histogram Analysis
```python
def calculate_histogram(image):
    """Calculate image histogram"""
    if len(image.shape) == 3:
        # Convert to grayscale
        gray = rgb_to_grayscale(image)
    else:
        gray = image
    
    hist, bins = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    return hist, bins

def histogram_equalization(image):
    """Apply histogram equalization"""
    # Calculate histogram
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
    
    # Calculate CDF
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    
    # Apply equalization
    equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    
    return equalized.reshape(image.shape).astype(np.uint8)
```

**NumPy Concepts:**
- `np.histogram()` for distribution analysis
- `flatten()` for 1D conversion
- `cumsum()` for cumulative sum
- `np.interp()` for interpolation

### Step 8: Geometric Transformations
```python
def rotate_image(image, angle):
    """Rotate image by 90-degree increments"""
    if angle == 90:
        return np.rot90(image, k=1)
    elif angle == 180:
        return np.rot90(image, k=2)
    elif angle == 270:
        return np.rot90(image, k=3)
    else:
        return image

def flip_image(image, direction='horizontal'):
    """Flip image horizontally or vertically"""
    if direction == 'horizontal':
        return np.fliplr(image)
    elif direction == 'vertical':
        return np.flipud(image)
    else:
        return image

def crop_image(image, x, y, width, height):
    """Crop image to specified region"""
    return image[y:y+height, x:x+width]
```

**NumPy Concepts:**
- `np.rot90()` for rotation
- `np.fliplr()` and `np.flipud()` for flipping
- Array slicing for cropping

### Step 9: Advanced Operations
```python
def blend_images(image1, image2, alpha=0.5):
    """Blend two images with alpha transparency"""
    # Ensure same size
    if image1.shape != image2.shape:
        raise ValueError("Images must have same dimensions")
    
    blended = alpha * image1.astype(np.float32) + (1 - alpha) * image2.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)

def threshold_image(image, threshold=128):
    """Apply binary threshold to image"""
    if len(image.shape) == 3:
        gray = rgb_to_grayscale(image)
    else:
        gray = image
    
    return np.where(gray > threshold, 255, 0).astype(np.uint8)

def gamma_correction(image, gamma=1.0):
    """Apply gamma correction"""
    # Normalize to 0-1 range
    normalized = image.astype(np.float32) / 255.0
    
    # Apply gamma correction
    corrected = np.power(normalized, 1.0 / gamma)
    
    # Convert back to 0-255 range
    return (corrected * 255).astype(np.uint8)
```

**NumPy Concepts:**
- `np.where()` for conditional operations
- `np.power()` for mathematical functions
- Array arithmetic with different data types

### Step 10: Complete Processing Pipeline
```python
def process_image_pipeline(image_path, operations):
    """Apply multiple operations in sequence"""
    # Load image
    image = np.array(Image.open(image_path))
    original = image.copy()
    
    results = {'original': original}
    
    for op_name, op_func, op_params in operations:
        try:
            image = op_func(image, **op_params)
            results[op_name] = image.copy()
            print(f"✅ Applied {op_name}")
        except Exception as e:
            print(f"❌ Error in {op_name}: {e}")
    
    return results
```

## 🎨 Sample Usage

### Basic Image Processing
```python
# Load and process image
image = np.array(Image.open('sample.jpg'))

# Apply transformations
gray = rgb_to_grayscale(image)
bright = adjust_brightness(image, 50)
contrast = adjust_contrast(image, 1.5)
blurred = apply_filter(image, create_blur_kernel(5))
edges = sobel_edge_detection(image)

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0,0].imshow(image); axes[0,0].set_title('Original')
axes[0,1].imshow(gray, cmap='gray'); axes[0,1].set_title('Grayscale')
axes[0,2].imshow(bright); axes[0,2].set_title('Brightened')
axes[1,0].imshow(contrast); axes[1,0].set_title('High Contrast')
axes[1,1].imshow(blurred); axes[1,1].set_title('Blurred')
axes[1,2].imshow(edges, cmap='gray'); axes[1,2].set_title('Edges')
plt.tight_layout()
plt.show()
```

### Processing Pipeline Example
```python
operations = [
    ('brightness', adjust_brightness, {'brightness': 30}),
    ('contrast', adjust_contrast, {'contrast': 1.2}),
    ('blur', lambda img, size: apply_filter(img, create_blur_kernel(size)), {'size': 3}),
    ('edges', sobel_edge_detection, {}),
]

results = process_image_pipeline('sample.jpg', operations)
```

## 📈 Performance Tips

### 1. Use Appropriate Data Types
```python
# Use float32 for calculations, uint8 for storage
image_float = image.astype(np.float32)
processed = process_image(image_float)
final = processed.astype(np.uint8)
```

### 2. Vectorize Operations
```python
# Avoid loops - use broadcasting
# Slow:
for i in range(height):
    for j in range(width):
        result[i,j] = image[i,j] * factor

# Fast:
result = image * factor
```

### 3. Memory Management
```python
# Use in-place operations when possible
image += brightness  # Instead of: image = image + brightness

# Delete large arrays when done
del large_array
```

## 🔍 Common Issues and Solutions

### Issue 1: Data Type Overflow
```python
# Problem: uint8 overflow
result = image + 100  # May overflow

# Solution: Use float32 and clip
result = np.clip(image.astype(np.float32) + 100, 0, 255).astype(np.uint8)
```

### Issue 2: Dimension Mismatch
```python
# Problem: Broadcasting error
image.shape  # (480, 640, 3)
kernel.shape  # (3, 3)

# Solution: Apply to each channel
for channel in range(3):
    filtered[:,:,channel] = convolve2d(image[:,:,channel], kernel)
```

### Issue 3: Memory Usage
```python
# Problem: Large images consume memory
# Solution: Process in chunks or use memory mapping
image_memmap = np.memmap('large_image.dat', dtype='uint8', mode='r', 
                        shape=(10000, 10000, 3))
```

## 🎯 Practice Exercises

1. **Create a sepia filter** using color transformation matrix
2. **Implement median filtering** for noise reduction
3. **Build a custom edge detection** algorithm
4. **Create artistic effects** like oil painting or watercolor
5. **Implement image morphology** operations (erosion, dilation)

## 📚 Next Steps

After mastering this project:
1. Move to Project 3: Financial Portfolio Analyzer
2. Explore OpenCV for advanced computer vision
3. Learn about deep learning for image processing
4. Try real-time image processing with webcam input

This comprehensive guide provides everything needed to master image processing with NumPy!
