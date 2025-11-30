import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Recognition",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2rem !important;
        font-weight: bold !important;
        text-align: center !important;
        color: #1f77b4 !important;
        margin-bottom: 0.5rem !important;
    }
    .sub-header {
        font-size: 1rem !important;
        text-align: center !important;
        color: #666 !important;
        margin-bottom: 1rem !important;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem 0;
    }
    .prediction-number {
        font-size: 3rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    /* Adjust main content width when sidebar is hidden */
    .main .block-container {
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: 1rem;
        max-width: 100%;
    }
    /* Reduce spacing in elements */
    .element-container {
        margin-bottom: 0.5rem;
    }
    h1, h2, h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Softmax Regression Model Class
class SoftmaxRegression:
    def __init__(self, n_features, n_classes):
        self.n_features = n_features
        self.n_classes = n_classes
        self.W = None
        self.b = None

    def softmax(self, z):
        """Softmax activation with numerical stability."""
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """Forward pass to compute predictions."""
        z = np.dot(X, self.W) + self.b
        return self.softmax(z)

    def predict(self, X):
        """Predict class labels."""
        y_pred_probs = self.forward(X)
        return np.argmax(y_pred_probs, axis=1), y_pred_probs

    def load_weights(self, filepath):
        """Load model weights from .npz file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        data = np.load(filepath)
        self.W = data['W']
        self.b = data['b']
        return True

@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try multiple possible paths
    possible_paths = [
        os.path.join(script_dir, '..', 'models', 'best_model_weights.npz'),
        os.path.join(script_dir, 'models', 'best_model_weights.npz'),
        '../models/best_model_weights.npz',
        'models/best_model_weights.npz',
        os.path.join(os.getcwd(), 'models', 'best_model_weights.npz'),
    ]
    
    model_path = None
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            model_path = abs_path
            break
    
    if model_path is None:
        raise FileNotFoundError(
            f"Model file not found. Tried paths: {possible_paths}\n"
            f"Current working directory: {os.getcwd()}\n"
            f"Script directory: {script_dir}"
        )
    
    model = SoftmaxRegression(n_features=784, n_classes=10)
    model.load_weights(model_path)
    return model

def preprocess_image(image):
    """
    Preprocess uploaded image to match MNIST format with sharpening.
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        Preprocessed image array (1, 784)
    """
    # Convert PIL to numpy array
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
        else:  # RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply threshold first to get sharper edges
    if len(np.unique(img_array)) > 2:
        # Use adaptive threshold for better edge preservation
        img_thresh = cv2.adaptiveThreshold(
            img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    else:
        img_thresh = img_array.copy()
    
    # Invert if needed (MNIST digits are white on black background)
    # Check if image is mostly dark (inverted)
    if np.mean(img_thresh) > 127:
        img_thresh = 255 - img_thresh
    
    # Resize with better interpolation for sharpness
    # Use INTER_CUBIC or INTER_LANCZOS4 for better quality when downscaling
    h, w = img_thresh.shape
    
    # If image is larger, use INTER_AREA (good for downscaling)
    # If image is smaller, use INTER_CUBIC (good for upscaling)
    if h > 28 or w > 28:
        img_resized = cv2.resize(img_thresh, (28, 28), interpolation=cv2.INTER_AREA)
    else:
        # For upscaling, use cubic interpolation for better quality
        img_resized = cv2.resize(img_thresh, (28, 28), interpolation=cv2.INTER_CUBIC)
    
    # Apply sharpening filter to enhance edges
    # Sharpening kernel
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    
    # Apply sharpening (convert to float for processing)
    img_float = img_resized.astype(np.float32)
    img_sharpened = cv2.filter2D(img_float, -1, sharpen_kernel)
    
    # Clip values to valid range [0, 255]
    img_sharpened = np.clip(img_sharpened, 0, 255).astype(np.uint8)
    
    # Optional: Apply morphological operations to clean up
    # Use opening to remove small noise
    kernel = np.ones((2, 2), np.uint8)
    img_cleaned = cv2.morphologyEx(img_sharpened, cv2.MORPH_CLOSE, kernel)
    
    # Normalize to [0, 1]
    img_normalized = img_cleaned.astype(np.float32) / 255.0
    
    # Flatten to (1, 784)
    img_flattened = img_normalized.reshape(1, -1)
    
    return img_flattened, img_normalized

def main():
    # Header
    st.markdown('<p class="main-header">MNIST Digit Recognition</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image of a handwritten digit (0-9) and get instant predictions</p>', unsafe_allow_html=True)
    
    # Load model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    

    # Upload file section (compact)
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image containing a handwritten digit"
    )
    
    if uploaded_file is not None:
        # Display images side by side
        image = Image.open(uploaded_file)
        
        # Preprocess image
        try:
            img_processed, img_normalized = preprocess_image(image)
        except Exception as e:
            st.error(f"Error preprocessing image: {str(e)}")
            st.stop()
        
        # Make prediction
        try:
            with st.spinner("Processing..."):
                prediction, probabilities = model.predict(img_processed)
                predicted_digit = prediction[0]
                confidence_scores = probabilities[0]
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.stop()
        
        # Three columns layout
        col1, col2, col3 = st.columns([1.2, 1.5, 2])
        
        # Column 1: Images (stacked vertically, smaller)
        with col1:
            st.markdown("**Images**")
            st.image(image, caption="Uploaded", width=150)
            fig_img = img_normalized.reshape(28, 28)
            st.image(fig_img, caption="Preprocessed (28×28)", width=150)
        
        # Column 2: Prediction and Top 3
        with col2:
            st.markdown("**Prediction**")
            # Display prediction with style
            st.markdown(f"""
            <div class="prediction-box">
                <h3 style="margin: 0.3rem 0; font-size: 1rem;">Predicted Digit</h3>
                <div class="prediction-number">{predicted_digit}</div>
                <p style="margin: 0.3rem 0; font-size: 0.9rem;">Confidence: {confidence_scores[predicted_digit]*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Top 3 predictions
            top3_indices = np.argsort(confidence_scores)[-3:][::-1]
            st.markdown("**Top 3 Predictions**")
            for rank, idx in enumerate(top3_indices, 1):
                st.markdown(f"**{rank}.** Digit **{idx}** ({confidence_scores[idx]*100:.2f}%)")
        
        # Column 3: Confidence Scores (compact)
        with col3:
            st.markdown("**Confidence Scores**")
            digits = list(range(10))
            confidences = [confidence_scores[i] * 100 for i in digits]
            
            # Display as compact bars
            for digit, conf in zip(digits, confidences):
                # Compact row with label, bar, and percentage
                cols = st.columns([1, 4, 1])
                with cols[0]:
                    if digit == predicted_digit:
                        st.markdown(f"<strong style='color: #667eea;'>D{digit}:</strong>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<small>D{digit}:</small>", unsafe_allow_html=True)
                with cols[1]:
                    st.progress(conf / 100.0)
                with cols[2]:
                    st.markdown(f"<small>{conf:.1f}%</small>", unsafe_allow_html=True)
    else:
        st.info("Please upload an image to get predictions")
    

if __name__ == "__main__":
    main()

