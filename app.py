"""
Advanced Image Processing Toolkit - Streamlit UI
Interactive interface for NumPy-based image processing
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import io
import base64

# Configure page
st.set_page_config(
    page_title="🖼️ Image Processing Toolkit",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        animation: slideInDown 1s ease-out;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    /* Animation keyframes */
    @keyframes slideInDown {
        from {
            opacity: 0;
            transform: translateY(-50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Image container styling */
    .image-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        animation: fadeInUp 0.8s ease-out;
        transition: transform 0.3s ease;
    }
    
    .image-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Control panel styling */
    .control-panel {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        animation: pulse 1s infinite;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
            font-size: 0.9rem;
        }
        .image-container {
            padding: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

class StreamlitImageProcessor:
    """Streamlit-based Image Processing Interface"""
    
    def __init__(self):
        if 'processed_images' not in st.session_state:
            st.session_state.processed_images = {}
        if 'current_image' not in st.session_state:
            st.session_state.current_image = None
        if 'image_history' not in st.session_state:
            st.session_state.image_history = []
    
    def generate_sample_image(self, image_type="gradient"):
        """Generate sample images for testing"""
        width, height = 400, 300
        
        if image_type == "gradient":
            # Create gradient image
            image_array = np.zeros((height, width, 3), dtype=np.uint8)
            for x in range(width):
                for y in range(height):
                    image_array[y, x, 0] = int(255 * x / width)  # Red gradient
                    image_array[y, x, 1] = int(255 * y / height)  # Green gradient
                    image_array[y, x, 2] = int(255 * (x + y) / (width + height))  # Blue gradient
        
        elif image_type == "shapes":
            # Create shapes image
            image_array = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Red rectangle
            image_array[50:150, 50:150, :] = [255, 0, 0]
            
            # Green circle (approximate)
            center_x, center_y = 275, 125
            radius = 75
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            image_array[mask] = [0, 255, 0]
            
            # Blue triangle (approximate)
            triangle_mask = np.zeros((height, width), dtype=bool)
            for y in range(200, 280):
                for x in range(100, 200):
                    if x >= 100 + (y - 200) * 0.5 and x <= 200 - (y - 200) * 0.5:
                        triangle_mask[y, x] = True
            image_array[triangle_mask] = [0, 0, 255]
        
        elif image_type == "noise":
            # Create noise image
            image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        return image_array
    
    def rgb_to_grayscale(self, image, method='luminance'):
        """Convert RGB to grayscale"""
        if len(image.shape) != 3:
            return image
        
        if method == 'luminance':
            return np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        elif method == 'average':
            return np.mean(image, axis=2).astype(np.uint8)
        else:
            return np.max(image, axis=2).astype(np.uint8)
    
    def adjust_brightness(self, image, brightness):
        """Adjust image brightness"""
        adjusted = image.astype(np.float32) + brightness
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def adjust_contrast(self, image, contrast):
        """Adjust image contrast"""
        adjusted = (image.astype(np.float32) - 128) * contrast + 128
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def gamma_correction(self, image, gamma):
        """Apply gamma correction"""
        normalized = image.astype(np.float32) / 255.0
        corrected = np.power(normalized, 1.0 / gamma)
        return (corrected * 255).astype(np.uint8)
    
    def create_blur_kernel(self, size=3, sigma=1.0):
        """Create Gaussian blur kernel"""
        kernel = np.zeros((size, size))
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
        
        return kernel / np.sum(kernel)
    
    def create_sharpen_kernel(self):
        """Create sharpening kernel"""
        return np.array([[ 0, -1,  0],
                        [-1,  5, -1],
                        [ 0, -1,  0]], dtype=np.float32)
    
    def apply_filter(self, image, kernel):
        """Apply convolution filter"""
        if len(image.shape) == 3:
            filtered = np.zeros_like(image, dtype=np.float32)
            for i in range(image.shape[2]):
                filtered[:, :, i] = convolve(image[:, :, i].astype(np.float32), kernel, mode='reflect')
            return np.clip(filtered, 0, 255).astype(np.uint8)
        else:
            filtered = convolve(image.astype(np.float32), kernel, mode='reflect')
            return np.clip(filtered, 0, 255).astype(np.uint8)
    
    def sobel_edge_detection(self, image):
        """Apply Sobel edge detection"""
        if len(image.shape) == 3:
            gray = self.rgb_to_grayscale(image)
        else:
            gray = image
        
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        edges_x = convolve(gray.astype(np.float32), sobel_x, mode='reflect')
        edges_y = convolve(gray.astype(np.float32), sobel_y, mode='reflect')
        
        edges = np.sqrt(edges_x**2 + edges_y**2)
        return np.clip(edges, 0, 255).astype(np.uint8)
    
    def create_sepia_effect(self, image):
        """Create sepia effect"""
        if len(image.shape) != 3:
            return image
        
        sepia_matrix = np.array([[0.393, 0.769, 0.189],
                                [0.349, 0.686, 0.168],
                                [0.272, 0.534, 0.131]])
        
        sepia = np.dot(image, sepia_matrix.T)
        return np.clip(sepia, 0, 255).astype(np.uint8)
    
    def threshold_image(self, image, threshold=128):
        """Apply binary threshold"""
        if len(image.shape) == 3:
            gray = self.rgb_to_grayscale(image)
        else:
            gray = image
        
        return np.where(gray > threshold, 255, 0).astype(np.uint8)
    
    def calculate_histogram(self, image):
        """Calculate image histogram"""
        if len(image.shape) == 3:
            gray = self.rgb_to_grayscale(image)
        else:
            gray = image
        
        hist, bins = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        return hist, bins
    
    def analyze_image(self, image):
        """Analyze image properties"""
        analysis = {
            'Shape': f"{image.shape[1]} × {image.shape[0]}",
            'Channels': image.shape[2] if len(image.shape) == 3 else 1,
            'Data Type': str(image.dtype),
            'Size (pixels)': image.size,
            'Mean Intensity': f"{np.mean(image):.1f}",
            'Std Deviation': f"{np.std(image):.1f}",
            'Min Value': int(np.min(image)),
            'Max Value': int(np.max(image))
        }
        
        if len(image.shape) == 3:
            analysis['Red Mean'] = f"{np.mean(image[:,:,0]):.1f}"
            analysis['Green Mean'] = f"{np.mean(image[:,:,1]):.1f}"
            analysis['Blue Mean'] = f"{np.mean(image[:,:,2]):.1f}"
        
        return analysis
    
    def create_histogram_plot(self, image):
        """Create histogram visualization"""
        hist, bins = self.calculate_histogram(image)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bins[:-1],
            y=hist,
            mode='lines',
            fill='tonexty',
            name='Histogram',
            line=dict(color='#667eea', width=2)
        ))
        
        fig.update_layout(
            title='Image Histogram',
            xaxis_title='Pixel Intensity',
            yaxis_title='Frequency',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333'),
            height=400
        )
        
        return fig
    
    def image_to_base64(self, image):
        """Convert image to base64 for download"""
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str

def main():
    # Initialize processor
    processor = StreamlitImageProcessor()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🖼️ Advanced Image Processing Toolkit</h1>
        <p>Interactive NumPy-powered image processing with real-time visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # Image source selection
    st.sidebar.markdown("### 📷 Image Source")
    image_source = st.sidebar.radio(
        "Choose image source:",
        ["Upload Image", "Generate Sample", "Use Camera"]
    )
    
    current_image = None
    
    if image_source == "Upload Image":
        uploaded_file = st.sidebar.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
        )
        
        if uploaded_file is not None:
            current_image = np.array(Image.open(uploaded_file))
            st.session_state.current_image = current_image
    
    elif image_source == "Generate Sample":
        sample_type = st.sidebar.selectbox(
            "Sample type:",
            ["gradient", "shapes", "noise"]
        )
        
        if st.sidebar.button("🎨 Generate Sample Image"):
            current_image = processor.generate_sample_image(sample_type)
            st.session_state.current_image = current_image
            st.success("✅ Sample image generated!")
    
    elif image_source == "Use Camera":
        camera_image = st.sidebar.camera_input("Take a picture")
        if camera_image is not None:
            current_image = np.array(Image.open(camera_image))
            st.session_state.current_image = current_image
    
    # Use session state image if available
    if st.session_state.current_image is not None:
        current_image = st.session_state.current_image
    
    if current_image is None:
        st.info("👆 Please select an image source from the sidebar to begin!")
        return
    
    # Processing options
    st.sidebar.markdown("### 🔧 Processing Options")
    
    # Basic adjustments
    st.sidebar.markdown("#### 💡 Basic Adjustments")
    brightness = st.sidebar.slider("Brightness", -100, 100, 0)
    contrast = st.sidebar.slider("Contrast", 0.1, 3.0, 1.0, 0.1)
    gamma = st.sidebar.slider("Gamma", 0.1, 3.0, 1.0, 0.1)
    
    # Filtering options
    st.sidebar.markdown("#### 🔍 Filters")
    apply_blur = st.sidebar.checkbox("Blur Filter")
    if apply_blur:
        blur_size = st.sidebar.slider("Blur Size", 3, 15, 5, 2)
        blur_sigma = st.sidebar.slider("Blur Sigma", 0.5, 5.0, 1.0, 0.5)
    
    apply_sharpen = st.sidebar.checkbox("Sharpen Filter")
    apply_edges = st.sidebar.checkbox("Edge Detection")
    
    # Effects
    st.sidebar.markdown("#### ✨ Effects")
    apply_grayscale = st.sidebar.checkbox("Grayscale")
    apply_sepia = st.sidebar.checkbox("Sepia")
    apply_negative = st.sidebar.checkbox("Negative")
    apply_threshold = st.sidebar.checkbox("Binary Threshold")
    if apply_threshold:
        threshold_value = st.sidebar.slider("Threshold", 0, 255, 128)
    
    # Process image
    processed_image = current_image.copy()
    
    # Apply adjustments
    if brightness != 0:
        processed_image = processor.adjust_brightness(processed_image, brightness)
    
    if contrast != 1.0:
        processed_image = processor.adjust_contrast(processed_image, contrast)
    
    if gamma != 1.0:
        processed_image = processor.gamma_correction(processed_image, gamma)
    
    # Apply filters
    if apply_blur:
        blur_kernel = processor.create_blur_kernel(blur_size, blur_sigma)
        processed_image = processor.apply_filter(processed_image, blur_kernel)
    
    if apply_sharpen:
        sharpen_kernel = processor.create_sharpen_kernel()
        processed_image = processor.apply_filter(processed_image, sharpen_kernel)
    
    if apply_edges:
        processed_image = processor.sobel_edge_detection(processed_image)
    
    # Apply effects
    if apply_grayscale:
        processed_image = processor.rgb_to_grayscale(processed_image)
    
    if apply_sepia:
        processed_image = processor.create_sepia_effect(processed_image)
    
    if apply_negative:
        processed_image = 255 - processed_image
    
    if apply_threshold:
        processed_image = processor.threshold_image(processed_image, threshold_value)
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="image-container">
            <h3 style="text-align: center; color: #333;">📷 Original Image</h3>
        </div>
        """, unsafe_allow_html=True)
        st.image(current_image, use_column_width=True)
        
        # Original image analysis
        with st.expander("📊 Original Image Analysis"):
            analysis = processor.analyze_image(current_image)
            for key, value in analysis.items():
                st.metric(key, value)
    
    with col2:
        st.markdown("""
        <div class="image-container">
            <h3 style="text-align: center; color: #333;">🎨 Processed Image</h3>
        </div>
        """, unsafe_allow_html=True)
        st.image(processed_image, use_column_width=True)
        
        # Processed image analysis
        with st.expander("📊 Processed Image Analysis"):
            processed_analysis = processor.analyze_image(processed_image)
            for key, value in processed_analysis.items():
                st.metric(key, value)
    
    # Histogram comparison
    st.markdown("## 📈 Histogram Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Original Histogram")
        original_hist_fig = processor.create_histogram_plot(current_image)
        st.plotly_chart(original_hist_fig, use_container_width=True)
    
    with col2:
        st.markdown("### Processed Histogram")
        processed_hist_fig = processor.create_histogram_plot(processed_image)
        st.plotly_chart(processed_hist_fig, use_container_width=True)
    
    # Download section
    st.markdown("## 💾 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download Original"):
            img_str = processor.image_to_base64(current_image)
            href = f'<a href="data:image/png;base64,{img_str}" download="original_image.png">Download Original Image</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        if st.button("📥 Download Processed"):
            img_str = processor.image_to_base64(processed_image)
            href = f'<a href="data:image/png;base64,{img_str}" download="processed_image.png">Download Processed Image</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col3:
        if st.button("🔄 Reset All"):
            st.experimental_rerun()
    
    # Quick presets
    st.markdown("## 🎯 Quick Presets")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🌅 Vintage"):
            st.session_state.preset = {
                'brightness': 20, 'contrast': 1.2, 'gamma': 0.8, 'sepia': True
            }
    
    with col2:
        if st.button("🔍 Sharp"):
            st.session_state.preset = {
                'contrast': 1.5, 'sharpen': True
            }
    
    with col3:
        if st.button("🌫️ Soft"):
            st.session_state.preset = {
                'blur': True, 'brightness': 10
            }
    
    with col4:
        if st.button("⚫ Dramatic"):
            st.session_state.preset = {
                'contrast': 2.0, 'gamma': 0.6, 'edges': True
            }
    
    # Advanced tools
    with st.expander("🔬 Advanced Tools"):
        st.markdown("### 🧮 NumPy Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.code(f"""
# Image Properties
Shape: {current_image.shape}
Data Type: {current_image.dtype}
Memory Usage: {current_image.nbytes / 1024:.1f} KB
            """)
        
        with col2:
            st.code(f"""
# Statistical Analysis
Mean: {np.mean(current_image):.2f}
Std Dev: {np.std(current_image):.2f}
Min/Max: {np.min(current_image)}/{np.max(current_image)}
            """)
        
        # Show processing pipeline
        st.markdown("### 🔄 Processing Pipeline")
        pipeline_steps = []
        
        if brightness != 0:
            pipeline_steps.append(f"Brightness: {brightness:+d}")
        if contrast != 1.0:
            pipeline_steps.append(f"Contrast: {contrast:.1f}x")
        if gamma != 1.0:
            pipeline_steps.append(f"Gamma: {gamma:.1f}")
        if apply_blur:
            pipeline_steps.append(f"Blur: {blur_size}px, σ={blur_sigma:.1f}")
        if apply_sharpen:
            pipeline_steps.append("Sharpen Filter")
        if apply_edges:
            pipeline_steps.append("Edge Detection")
        if apply_grayscale:
            pipeline_steps.append("Grayscale Conversion")
        if apply_sepia:
            pipeline_steps.append("Sepia Effect")
        if apply_negative:
            pipeline_steps.append("Negative Effect")
        if apply_threshold:
            pipeline_steps.append(f"Threshold: {threshold_value}")
        
        if pipeline_steps:
            for i, step in enumerate(pipeline_steps, 1):
                st.write(f"{i}. {step}")
        else:
            st.write("No processing applied")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🌟 Built with Streamlit, NumPy & Plotly | Image Processing Toolkit</p>
        <p>Real-time Processing • Interactive Controls • Advanced Analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()