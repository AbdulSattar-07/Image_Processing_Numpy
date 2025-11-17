# Project 2: Image Processing - Formulas & Detailed Concepts

## 📐 Mathematical Formulas for Image Processing

### 1. Color Space Conversions

#### RGB to Grayscale (Luminance Method)
```
Grayscale = 0.299 × R + 0.587 × G + 0.114 × B
```
**Explanation:**
- Uses human eye sensitivity to different colors
- Green contributes most (58.7%), red moderate (29.9%), blue least (11.4%)
- **NumPy Implementation:**
```python
grayscale = np.dot(image[...,:3], [0.299, 0.587, 0.114])
```

#### RGB to Grayscale (Average Method)
```
Grayscale = (R + G + B) / 3
```
**NumPy Implementation:**
```python
grayscale = np.mean(image, axis=2)
```

#### RGB to HSV Conversion
```
V = max(R, G, B)
S = (V - min(R, G, B)) / V  (if V ≠ 0)
H = depends on which color is maximum
```

### 2. Intensity Transformations

#### Linear Brightness Adjustment
```
New_Pixel = Original_Pixel + Brightness_Offset
```
**Range:** Clamp result to [0, 255]
**NumPy Implementation:**
```python
bright_image = np.clip(image + brightness, 0, 255)
```

#### Linear Contrast Adjustment
```
New_Pixel = (Original_Pixel - 128) × Contrast_Factor + 128
```
**Explanation:**
- Subtract 128 to center around middle gray
- Multiply by contrast factor
- Add 128 back to restore center
**NumPy Implementation:**
```python
contrast_image = np.clip((image - 128) * contrast + 128, 0, 255)
```

#### Gamma Correction
```
New_Pixel = 255 × (Original_Pixel / 255)^(1/gamma)
```
**Gamma Values:**
- γ < 1: Brightens image (expands dark regions)
- γ > 1: Darkens image (compresses dark regions)
- γ = 1: No change
**NumPy Implementation:**
```python
gamma_corrected = 255 * np.power(image/255.0, 1/gamma)
```

### 3. Histogram Operations

#### Histogram Calculation
```
Histogram[i] = Number of pixels with intensity value i
```
**NumPy Implementation:**
```python
hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
```

#### Cumulative Distribution Function (CDF)
```
CDF[i] = Σ(j=0 to i) Histogram[j]
```
**NumPy Implementation:**
```python
cdf = np.cumsum(hist)
```

#### Histogram Equalization
```
New_Pixel = (CDF[Original_Pixel] × 255) / Total_Pixels
```
**Purpose:** Spreads pixel intensities across full range
**NumPy Implementation:**
```python
equalized = ((cdf[image] - cdf.min()) * 255 / (cdf.max() - cdf.min())).astype(np.uint8)
```

### 4. Convolution and Filtering

#### 2D Convolution Formula
```
Output[i,j] = Σ Σ Image[i+m, j+n] × Kernel[m, n]
              m n
```
**Explanation:**
- Slide kernel over image
- Multiply corresponding elements
- Sum all products for output pixel

#### Gaussian Blur Kernel
```
G(x,y) = (1/(2πσ²)) × e^(-(x²+y²)/(2σ²))
```
**Parameters:**
- σ (sigma): Controls blur amount
- Larger σ = more blur
**3x3 Gaussian Kernel Example:**
```
[1, 2, 1]
[2, 4, 2] × (1/16)
[1, 2, 1]
```

#### Box Blur Kernel
```
All elements = 1 / (kernel_size²)
```
**3x3 Box Blur:**
```
[1, 1, 1]
[1, 1, 1] × (1/9)
[1, 1, 1]
```

### 5. Edge Detection

#### Sobel Operator
**Horizontal Edge Detection (Gx):**
```
[-1, 0, 1]
[-2, 0, 2]
[-1, 0, 1]
```

**Vertical Edge Detection (Gy):**
```
[-1, -2, -1]
[ 0,  0,  0]
[ 1,  2,  1]
```

**Edge Magnitude:**
```
Magnitude = √(Gx² + Gy²)
```

**Edge Direction:**
```
Direction = arctan(Gy / Gx)
```

#### Laplacian Edge Detection
```
[ 0, -1,  0]
[-1,  4, -1]
[ 0, -1,  0]
```
**Purpose:** Detects edges in all directions

#### Prewitt Operator
**Horizontal:**
```
[-1, 0, 1]
[-1, 0, 1]
[-1, 0, 1]
```

**Vertical:**
```
[-1, -1, -1]
[ 0,  0,  0]
[ 1,  1,  1]
```

### 6. Sharpening Filters

#### Unsharp Masking
```
Sharpened = Original + α × (Original - Blurred)
```
**Where α is sharpening strength**

#### Sharpen Kernel
```
[ 0, -1,  0]
[-1,  5, -1]
[ 0, -1,  0]
```

### 7. Morphological Operations

#### Erosion
```
Output[i,j] = min(Image[i+m, j+n] + StructuringElement[m,n])
```

#### Dilation
```
Output[i,j] = max(Image[i+m, j+n] + StructuringElement[m,n])
```

## 🧮 Detailed NumPy Concepts

### 1. Multi-dimensional Array Indexing

#### 3D Array Structure
```python
image.shape  # (height, width, channels)
# For RGB image: (H, W, 3)
# For RGBA image: (H, W, 4)
# For grayscale: (H, W) or (H, W, 1)
```

#### Channel Access
```python
red_channel = image[:, :, 0]      # Red channel
green_channel = image[:, :, 1]    # Green channel  
blue_channel = image[:, :, 2]     # Blue channel
all_channels = image[:, :, :]     # All channels
```

#### Spatial Slicing
```python
# Crop image
cropped = image[y1:y2, x1:x2, :]

# Get image quadrants
top_left = image[:h//2, :w//2, :]
top_right = image[:h//2, w//2:, :]
bottom_left = image[h//2:, :w//2, :]
bottom_right = image[h//2:, w//2:, :]
```

### 2. Broadcasting in Image Processing

#### Scalar Broadcasting
```python
# Add brightness to all pixels
brighter = image + 50

# Multiply all channels
darker = image * 0.5

# Apply to specific channel
image[:, :, 0] = image[:, :, 0] * 1.2  # Enhance red
```

#### Array Broadcasting
```python
# Apply different values to each channel
channel_multipliers = np.array([1.2, 1.0, 0.8])  # R, G, B
adjusted = image * channel_multipliers
```

### 3. Conditional Operations

#### Thresholding
```python
# Binary threshold
binary = np.where(grayscale > threshold, 255, 0)

# Multi-level threshold
result = np.select([grayscale < 85, grayscale < 170], [0, 128], 255)
```

#### Masking
```python
# Create mask
mask = (image[:, :, 0] > 100) & (image[:, :, 1] < 50)

# Apply mask
masked_image = np.where(mask[..., np.newaxis], image, 0)
```

### 4. Array Manipulation for Images

#### Reshaping
```python
# Flatten image to pixel list
pixels = image.reshape(-1, 3)  # Shape: (height*width, 3)

# Reshape back to image
restored = pixels.reshape(height, width, 3)
```

#### Transposition
```python
# Swap axes
transposed = np.transpose(image, (1, 0, 2))  # Swap height/width

# Channel-first format (for deep learning)
chw_format = np.transpose(image, (2, 0, 1))  # (C, H, W)
```

#### Concatenation
```python
# Horizontal concatenation
combined_h = np.concatenate([image1, image2], axis=1)

# Vertical concatenation  
combined_v = np.concatenate([image1, image2], axis=0)

# Channel concatenation
rgba = np.concatenate([rgb_image, alpha_channel[..., np.newaxis]], axis=2)
```

### 5. Statistical Operations

#### Per-Channel Statistics
```python
# Mean per channel
channel_means = np.mean(image, axis=(0, 1))  # Shape: (3,)

# Standard deviation per channel
channel_stds = np.std(image, axis=(0, 1))

# Min/Max per channel
channel_mins = np.min(image, axis=(0, 1))
channel_maxs = np.max(image, axis=(0, 1))
```

#### Spatial Statistics
```python
# Mean across channels
spatial_mean = np.mean(image, axis=2)  # Shape: (H, W)

# Variance across channels
spatial_var = np.var(image, axis=2)
```

### 6. Advanced Array Operations

#### Rolling/Shifting
```python
# Shift image
shifted = np.roll(image, shift=(10, 20), axis=(0, 1))

# Circular shift per channel
shifted_channels = np.roll(image, shift=1, axis=2)
```

#### Padding
```python
# Pad image for convolution
padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='reflect')

# Zero padding
zero_padded = np.pad(image, ((10, 10), (10, 10), (0, 0)), mode='constant')
```

#### Interpolation
```python
# Nearest neighbor upsampling
upsampled = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)

# Simple downsampling
downsampled = image[::2, ::2, :]
```

## 🔧 Custom Filter Implementation

### Convolution Function
```python
def apply_convolution(image, kernel):
    """Apply 2D convolution to grayscale image"""
    h, w = image.shape
    kh, kw = kernel.shape
    
    # Pad image
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    
    # Initialize output
    output = np.zeros_like(image)
    
    # Apply convolution
    for i in range(h):
        for j in range(w):
            output[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    
    return output
```

### Vectorized Convolution
```python
def vectorized_convolution(image, kernel):
    """Faster vectorized convolution"""
    from scipy.ndimage import convolve
    return convolve(image, kernel, mode='reflect')
```

## 📊 Performance Optimization

### Memory Efficient Operations
```python
# In-place operations
image += brightness  # Instead of: image = image + brightness

# Use views when possible
red_view = image[:, :, 0]  # View, not copy
red_copy = image[:, :, 0].copy()  # Explicit copy
```

### Vectorization Tips
```python
# Avoid loops - use broadcasting
# Slow:
for i in range(height):
    for j in range(width):
        result[i, j] = image[i, j] * factor

# Fast:
result = image * factor
```

### Data Type Considerations
```python
# Use appropriate data types
image_float = image.astype(np.float32)  # For calculations
image_uint8 = np.clip(result, 0, 255).astype(np.uint8)  # For display
```

## 🎯 Common Patterns

### Image Processing Pipeline
```python
def process_image(image):
    # Convert to float for processing
    img_float = image.astype(np.float32)
    
    # Apply operations
    processed = apply_brightness(img_float, brightness)
    processed = apply_contrast(processed, contrast)
    processed = apply_filter(processed, kernel)
    
    # Convert back to uint8
    return np.clip(processed, 0, 255).astype(np.uint8)
```

### Error Handling
```python
def safe_image_operation(image, operation):
    try:
        # Validate input
        assert image.ndim in [2, 3], "Image must be 2D or 3D"
        assert image.dtype == np.uint8, "Image must be uint8"
        
        # Apply operation
        result = operation(image)
        
        # Validate output
        return np.clip(result, 0, 255).astype(np.uint8)
    
    except Exception as e:
        print(f"Error in image operation: {e}")
        return image  # Return original on error
```

Ye comprehensive guide aapko image processing ke saare mathematical formulas aur NumPy concepts ki deep understanding dega!