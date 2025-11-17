# Project 2: Image Processing Toolkit - Complete Guide

## 🖼️ Project Overview
Build a comprehensive image processing toolkit using NumPy to manipulate images as multi-dimensional arrays. This project teaches advanced NumPy concepts through practical computer vision applications.

## 🎯 Learning Objectives
By completing this project, you will master:
- Working with 3D arrays (height, width, RGB channels)
- Array broadcasting for efficient operations
- Mathematical operations on multi-dimensional arrays
- Array reshaping and transposition
- Masking and conditional operations
- Image filtering and convolution
- Color space transformations

## 📊 Mathematical Formulas Used

### 1. Grayscale Conversion
```
Grayscale = 0.299 × R + 0.587 × G + 0.114 × B
```
**Purpose:** Convert color image to grayscale using luminance weights
**NumPy:** `np.dot(image, [0.299, 0.587, 0.114])`

### 2. Brightness Adjustment
```
New_Pixel = Original_Pixel + Brightness_Value
```
**Purpose:** Increase or decrease image brightness
**NumPy:** `image + brightness_factor`

### 3. Contrast Adjustment
```
New_Pixel = (Original_Pixel - 128) × Contrast_Factor + 128
```
**Purpose:** Enhance or reduce image contrast
**NumPy:** `(image - 128) * contrast_factor + 128`

### 4. Gamma Correction
```
New_Pixel = 255 × (Original_Pixel / 255)^(1/gamma)
```
**Purpose:** Adjust image gamma for display correction
**NumPy:** `255 * np.power(image/255, 1/gamma)`

### 5. Gaussian Blur Kernel
```
G(x,y) = (1/(2πσ²)) × e^(-(x²+y²)/(2σ²))
```
**Purpose:** Create smooth blur effect
**NumPy:** Custom kernel generation and convolution

### 6. Sobel Edge Detection
```
Gx = [-1, 0, 1; -2, 0, 2; -1, 0, 1]
Gy = [-1, -2, -1; 0, 0, 0; 1, 2, 1]
Magnitude = √(Gx² + Gy²)
```
**Purpose:** Detect edges in images
**NumPy:** Convolution with Sobel kernels

### 7. Image Histogram
```
Histogram[i] = Count of pixels with intensity i
```
**Purpose:** Analyze pixel intensity distribution
**NumPy:** `np.histogram(image, bins=256)`

### 8. Histogram Equalization
```
CDF[i] = Σ(Histogram[0] to Histogram[i])
New_Pixel = (CDF[pixel] × 255) / total_pixels
```
**Purpose:** Improve image contrast
**NumPy:** Cumulative distribution function

## 🔄 Complete Project Steps

### Step 1: Image Loading and Representation
```python
# Load image as NumPy array
image = np.array(PIL.Image.open('image.jpg'))
print(f"Image shape: {image.shape}")  # (height, width, channels)
```

### Step 2: Basic Image Properties
```python
# Analyze image properties
height, width, channels = image.shape
total_pixels = height * width
print(f"Resolution: {width}x{height}")
print(f"Color channels: {channels}")
```

### Step 3: Grayscale Conversion
```python
# Convert RGB to grayscale
grayscale = np.dot(image[...,:3], [0.299, 0.587, 0.114])
```

### Step 4: Brightness and Contrast
```python
# Adjust brightness
bright_image = np.clip(image + brightness, 0, 255)

# Adjust contrast
contrast_image = np.clip((image - 128) * contrast + 128, 0, 255)
```

### Step 5: Color Channel Manipulation
```python
# Split color channels
red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]

# Create color-filtered images
red_only = np.zeros_like(image)
red_only[:, :, 0] = red_channel
```

### Step 6: Image Filtering
```python
# Apply blur filter
blurred = apply_filter(image, blur_kernel)

# Apply sharpen filter
sharpened = apply_filter(image, sharpen_kernel)
```

### Step 7: Edge Detection
```python
# Sobel edge detection
edges_x = apply_filter(grayscale, sobel_x_kernel)
edges_y = apply_filter(grayscale, sobel_y_kernel)
edges = np.sqrt(edges_x**2 + edges_y**2)
```

### Step 8: Histogram Analysis
```python
# Calculate histogram
hist, bins = np.histogram(grayscale, bins=256, range=(0, 256))

# Histogram equalization
equalized = histogram_equalization(grayscale)
```

### Step 9: Image Transformations
```python
# Rotate image
rotated = np.rot90(image, k=1)

# Flip image
flipped_h = np.fliplr(image)
flipped_v = np.flipud(image)
```

### Step 10: Advanced Operations
```python
# Image blending
blended = alpha * image1 + (1 - alpha) * image2

# Threshold operations
binary = np.where(grayscale > threshold, 255, 0)
```

## 🧮 NumPy Concepts Covered

### 1. Multi-dimensional Array Operations
```python
# 3D array manipulation
image.shape                    # (height, width, channels)
image[:, :, 0]                # Red channel
image[100:200, 50:150, :]     # Image crop
```

### 2. Broadcasting with Images
```python
# Apply operation to all pixels
brightened = image + 50        # Add to all channels
normalized = image / 255.0     # Normalize to 0-1 range
```

### 3. Array Slicing for Image Processing
```python
# Advanced slicing
top_half = image[:height//2, :, :]
left_half = image[:, :width//2, :]
every_other = image[::2, ::2, :]  # Downsample
```

### 4. Conditional Operations
```python
# Masking operations
mask = grayscale > 128
result = np.where(mask, 255, 0)  # Binary threshold
```

### 5. Mathematical Operations
```python
# Element-wise operations
squared = np.square(image)
sqrt_image = np.sqrt(image)
log_image = np.log(image + 1)
```

### 6. Array Reshaping for Images
```python
# Reshape operations
flattened = image.reshape(-1, 3)  # Flatten to pixel list
reshaped = image.reshape(new_height, new_width, 3)
```

### 7. Convolution Operations
```python
# Custom convolution implementation
def convolve2d(image, kernel):
    return scipy.ndimage.convolve(image, kernel)
```

### 8. Statistical Analysis
```python
# Image statistics
mean_intensity = np.mean(image)
std_intensity = np.std(image)
min_max = np.min(image), np.max(image)
```

## 🎨 Image Processing Techniques

### 1. Color Space Conversions
- RGB to Grayscale
- RGB to HSV
- Color channel separation
- Color filtering

### 2. Intensity Adjustments
- Brightness control
- Contrast enhancement
- Gamma correction
- Histogram equalization

### 3. Filtering Operations
- Blur filters (Gaussian, Box)
- Sharpen filters
- Edge detection (Sobel, Laplacian)
- Custom kernel convolution

### 4. Geometric Transformations
- Image rotation
- Horizontal/vertical flipping
- Cropping and resizing
- Image translation

### 5. Advanced Techniques
- Image blending
- Thresholding
- Morphological operations
- Noise reduction

## 📈 Learning Progression

### Beginner Level (Days 1-3)
- Image loading and basic properties
- Grayscale conversion
- Simple brightness/contrast adjustments
- Color channel manipulation

### Intermediate Level (Days 4-7)
- Image filtering and convolution
- Edge detection algorithms
- Histogram analysis
- Geometric transformations

### Advanced Level (Days 8-10)
- Custom filter design
- Advanced color space operations
- Performance optimization
- Real-world image processing

## 🔍 Practical Applications

### Photography Enhancement
- Automatic brightness/contrast adjustment
- Color correction and filtering
- Noise reduction techniques
- Artistic effect creation

### Computer Vision Preprocessing
- Edge detection for object recognition
- Image normalization for ML models
- Feature extraction techniques
- Image segmentation preparation

### Medical Image Analysis
- X-ray and MRI enhancement
- Contrast improvement for diagnosis
- Edge detection for structure analysis
- Histogram analysis for abnormality detection

### Scientific Image Analysis
- Microscopy image enhancement
- Satellite image processing
- Astronomical image analysis
- Material science imaging

## 💡 Key Takeaways

1. **Images are 3D NumPy arrays** with shape (height, width, channels)
2. **Broadcasting enables efficient** pixel-wise operations
3. **Convolution is fundamental** to image filtering
4. **Array slicing provides** powerful image manipulation
5. **Mathematical operations** create various visual effects
6. **Histogram analysis reveals** image characteristics
7. **Vectorization is crucial** for performance in image processing

## 🛠️ Tools and Libraries

### Core Libraries
- **NumPy** - Array operations and mathematical functions
- **PIL/Pillow** - Image loading and saving
- **Matplotlib** - Image display and visualization
- **SciPy** - Advanced filtering operations

### Optional Enhancements
- **OpenCV** - Advanced computer vision
- **scikit-image** - Scientific image processing
- **ImageIO** - Various image format support

## 📝 Project Structure
```
project2_image_processing/
├── project2_image_processing.py      # Main implementation
├── project2_formulas_details.md      # Detailed formulas
├── project2_README.md               # Step-by-step guide
├── sample_images/                   # Test images
├── processed_images/                # Output images
└── image_processing_streamlit.py    # Interactive UI
```

This comprehensive guide provides everything needed to master image processing with NumPy through hands-on practical experience!