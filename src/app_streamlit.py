import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from data_prep import test_generator
import os
import cv2
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="X-Ray Diagnostics AI", layout="wide", initial_sidebar_state="collapsed")

# Sidebar
st.sidebar.title("⚙️ Settings")
if st.sidebar.button("🌓 Toggle Theme"):
    st.session_state.dark_mode = not getattr(st.session_state, 'dark_mode', True)
dark_mode = st.session_state.get('dark_mode', True)

# Premium CSS
st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
        
        * {{ font-family: 'Poppins', sans-serif; }}
        #MainMenu, footer, header {{ visibility: hidden; }}
        
        .stApp {{
            {'background: linear-gradient(to bottom right, #f8fafc, #e0e7ff, #dbeafe);' if not dark_mode else 'background: linear-gradient(to bottom right, #0f172a, #1e1b4b, #1e293b);'}
            color: {'#1e293b' if not dark_mode else '#f1f5f9'};
        }}
        
        .hero {{
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%);
            padding: 3.5rem 2rem;
            border-radius: 24px;
            box-shadow: 0 20px 60px rgba(79, 70, 229, 0.4);
            margin-bottom: 3rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        
        .hero::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: pulse 15s ease-in-out infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1) rotate(0deg); }}
            50% {{ transform: scale(1.1) rotate(180deg); }}
        }}
        
        .hero h1 {{
            color: white;
            font-size: 3rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -1px;
            text-shadow: 0 4px 20px rgba(0,0,0,0.3);
            position: relative;
            z-index: 1;
        }}
        
        .hero p {{
            color: rgba(255, 255, 255, 0.95);
            font-size: 1.25rem;
            margin-top: 1rem;
            font-weight: 400;
            position: relative;
            z-index: 1;
        }}
        
        .info-card {{
            background: {'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)' if not dark_mode else 'linear-gradient(135deg, #1e293b 0%, #334155 100%)'};
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 8px 32px {'rgba(0, 0, 0, 0.08)' if not dark_mode else 'rgba(0, 0, 0, 0.3)'};
            border: 1px solid {'rgba(148, 163, 184, 0.2)' if not dark_mode else 'rgba(148, 163, 184, 0.1)'};
            margin: 2rem 0;
            backdrop-filter: blur(10px);
        }}
        
        .info-card h3 {{
            color: {'#1e293b' if not dark_mode else '#f1f5f9'};
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
        }}
        
        .condition-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }}
        
        .condition-item {{
            background: {'#f1f5f9' if not dark_mode else '#0f172a'};
            padding: 1.25rem;
            border-radius: 12px;
            border-left: 4px solid #4f46e5;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .condition-item:hover {{
            transform: translateX(4px);
            box-shadow: 0 4px 20px rgba(79, 70, 229, 0.2);
        }}
        
        .condition-item strong {{
            color: #4f46e5;
            font-weight: 600;
            font-size: 1.05rem;
        }}
        
        .metric-card {{
            background: {'linear-gradient(135deg, #ffffff 0%, #fefce8 100%)' if not dark_mode else 'linear-gradient(135deg, #1e293b 0%, #312e81 100%)'};
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 8px 32px {'rgba(79, 70, 229, 0.15)' if not dark_mode else 'rgba(0, 0, 0, 0.3)'};
            border: 2px solid {'rgba(79, 70, 229, 0.2)' if not dark_mode else 'rgba(79, 70, 229, 0.3)'};
            text-align: center;
            transition: all 0.3s;
            margin: 1rem 0;
        }}
        
        .metric-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 48px {'rgba(79, 70, 229, 0.25)' if not dark_mode else 'rgba(0, 0, 0, 0.5)'};
        }}
        
        .metric-label {{
            font-size: 0.95rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: {'#64748b' if not dark_mode else '#94a3b8'};
            margin-bottom: 0.75rem;
        }}
        
        .metric-value {{
            font-size: 2.75rem;
            font-weight: 800;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0.5rem 0;
        }}
        
        .metric-change {{
            font-size: 0.9rem;
            color: #10b981;
            font-weight: 600;
            margin-top: 0.5rem;
        }}
        
        .result-card {{
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            padding: 2.5rem;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(79, 70, 229, 0.4);
            text-align: center;
            color: white;
            margin: 2rem 0;
        }}
        
        .result-card .label {{
            font-size: 1rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 2px;
            opacity: 0.9;
        }}
        
        .result-card .prediction {{
            font-size: 3.5rem;
            font-weight: 900;
            margin: 1rem 0;
            text-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }}
        
        .result-card .details {{
            font-size: 1.1rem;
            font-weight: 500;
            opacity: 0.95;
        }}
        
        .stButton > button {{
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            border: none;
            padding: 1rem 3rem;
            border-radius: 12px;
            font-weight: 700;
            font-size: 1.1rem;
            transition: all 0.3s;
            box-shadow: 0 8px 24px rgba(79, 70, 229, 0.4);
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 12px 36px rgba(79, 70, 229, 0.5);
        }}
        
        .stProgress > div > div > div > div {{ 
            background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        }}
        
        .upload-section {{
            background: {'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)' if not dark_mode else 'linear-gradient(135deg, #1e293b 0%, #334155 100%)'};
            padding: 3rem;
            border-radius: 20px;
            box-shadow: 0 8px 32px {'rgba(0, 0, 0, 0.08)' if not dark_mode else 'rgba(0, 0, 0, 0.3)'};
            text-align: center;
            border: 2px dashed {'#cbd5e1' if not dark_mode else '#475569'};
            margin: 2rem 0;
        }}
        
        .footer {{
            text-align: center;
            color: #64748b;
            margin-top: 4rem;
            padding: 2rem;
            font-size: 0.9rem;
            font-weight: 500;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Hero Header
st.markdown("""
<div class="hero">
    <h1>🩺 AI Chest X-Ray Diagnostics</h1>
    <p>Advanced Multi-Disease Detection System</p>
</div>
""", unsafe_allow_html=True)

# Info Card
st.markdown("""
<div class="info-card">
    <h3>📋 Diagnostic Capabilities</h3>
    <p>Our AI-powered system analyzes chest X-rays to detect multiple pulmonary conditions with high precision:</p>
    <div class="condition-grid">
        <div class="condition-item">
            <strong>🫁 Pneumonia</strong><br>
            Acute lung inflammation from infection
        </div>
        <div class="condition-item">
            <strong>🦠 Tuberculosis</strong><br>
            Bacterial infection of the lungs
        </div>
        <div class="condition-item">
            <strong>🎗️ Lung Cancer</strong><br>
            Malignant pulmonary growth
        </div>
        <div class="condition-item">
            <strong>✅ Normal</strong><br>
            Healthy lung tissue detected
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Upload Section
uploaded_file = st.file_uploader("📤 Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

# Model configuration
model_paths = [
    os.path.join('saved_models', 'custom_cnn.h5'),
    os.path.join('saved_models', 'vgg16.h5'),
    os.path.join('saved_models', 'resnet50.h5'),
    os.path.join('saved_models', 'efficientnet.h5')
]
model_names = ['Custom CNN', 'VGG16', 'ResNet50', 'EfficientNet']

if uploaded_file is not None:
    img_path = "temp_uploaded_image.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("🔬 Analyzing X-Ray with AI Models..."):
        # Load and preprocess
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Model evaluation
        st.markdown("### 📊 Model Performance Metrics")
        cols = st.columns(4)
        accuracies = {}
        for idx, (model_path, name) in enumerate(zip(model_paths, model_names)):  # Fixed enumeration
            try:
                model = load_model(model_path)
                loss, acc = model.evaluate(test_generator)
                accuracies[name] = acc
                with cols[idx]:
                    st.markdown(f"<div class='metric-card'><div class='metric-label'>{name}</div><div class='metric-value'>{acc:.2%}</div></div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading {name}: {str(e)}")

        # Predictions with debug
        predictions = {}
        for model_path, name in zip(model_paths, model_names):
            try:
                model = load_model(model_path)
                pred = model.predict(img_array)
                probabilities = pred[0] * 100  # Convert to percentage
                print(f"Model: {name}, Probabilities: {probabilities}")  # Debug print
                predictions[name] = {"pred": pred, "probability": np.max(probabilities)}
            except Exception as e:
                st.error(f"Error predicting with {name}: {str(e)}")

        # Find the best prediction (highest probability)
        best_name = max(predictions, key=lambda x: predictions[x]["probability"])
        best_pred = predictions[best_name]["pred"]
        best_probability = predictions[best_name]["probability"]
        
        class_names = ['LUNG_CANCER', 'NORMAL', 'PNEUMONIA', 'TB']
        disease_names = {
            'LUNG_CANCER': 'Lung Cancer',
            'NORMAL': 'Normal (Healthy)',
            'PNEUMONIA': 'Pneumonia',
            'TB': 'Tuberculosis'
        }
        predicted_class = class_names[np.argmax(best_pred[0])]
        disease_description = disease_names[predicted_class]

    # Display results
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption="Uploaded Chest X-Ray", use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div class="result-card">
            <div class="label">Diagnosis Result</div>
            <div class="prediction">{predicted_class}</div>
            <div class="details">
                {disease_description}<br>
                Confidence: {best_probability:.1f}%<br>
                Model Used: {best_name}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Heatmap generation
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🔥 Generate Activation Heatmap"):
            # Simple heatmap based on prediction probabilities
            heatmap = cv2.applyColorMap(np.uint8(255 * best_pred[0]), cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
            overlay = cv2.addWeighted(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
            cv2.imwrite('heatmap_output.jpg', overlay)
            st.image('heatmap_output.jpg', caption="AI Activation Heatmap", use_container_width=True)
            if os.path.exists('heatmap_output.jpg'):
                os.remove('heatmap_output.jpg')

    # Cleanup
    if os.path.exists(img_path):
        os.remove(img_path)

# Footer
st.markdown("""
<div class="footer">
    <strong>AI Diagnostics System</strong> • Powered by ANN • © 2025
</div>
""", unsafe_allow_html=True)