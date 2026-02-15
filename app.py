import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import io

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="AutoInspect AI - Intelligent Damage Assessment",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner, professional look
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    h1, h2, h3 { color: #2c3e50; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGIC & HELPER FUNCTIONS ---

def calculate_severity(box_area, image_area, confidence):
    """
    Heuristic to determine severity based on damage size relative to vehicle size
    and model confidence.
    """
    ratio = box_area / image_area
    
    if ratio < 0.02: # Less than 2% of the image
        severity = "Low"
    elif ratio < 0.10: # Between 2% and 10%
        severity = "Medium"
    else: # Greater than 10%
        severity = "High"
        
    return severity

def categorize_damage_type(label):
    """Classifies damage into Cosmetic vs. Functional."""
    functional_keywords = ['crack', 'broken', 'missing', 'glass', 'lamp']
    label_lower = label.lower()
    
    if any(k in label_lower for k in functional_keywords):
        return "Functional (Critical)"
    return "Cosmetic (Surface)"

@st.cache_resource
def load_model(model_path):
    """Loads and caches the YOLO model."""
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 3. MAIN APPLICATION ---

def main():
    # --- Sidebar: Controls ---
    with st.sidebar:
        st.title("⚙️ Inspection Settings")
        st.info("Upload a clear image of the vehicle.")
        
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 
                                   help="Minimum probability to consider a detection valid.")
        
        # Placeholder for model selection if you have multiple versions
        model_version = "best.pt" 
        
    # --- Main Content ---
    st.title("🚗 AutoInspect AI")
    st.markdown("#### Intelligent Vehicle Damage Detection & Assessment System")
    st.write("Upload an image to generate a standardized damage report for insurance or repair estimation.")
    
    # Model Loading
    model = load_model(model_version)
    if not model:
        st.stop()

    # File Uploader
    uploaded_file = st.file_uploader("Drop vehicle image here...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            # Load and Preprocess
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)
            height, width, _ = img_array.shape
            image_area = height * width

            # --- DETECTION PHASE ---
            with st.spinner("Analyzing vehicle surface..."):
                results = model.predict(img_array, conf=conf_threshold)

            # --- RESULTS PROCESSING ---
            detected_objects = []
            
            # Extract data from YOLO results
            if results and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    # Bounding Box Coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    box_w = x2 - x1
                    box_h = y2 - y1
                    box_area = box_w * box_h
                    
                    # Metadata
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls]
                    
                    # Business Logic
                    severity = calculate_severity(box_area, image_area, conf)
                    category = categorize_damage_type(label)
                    
                    detected_objects.append({
                        "Type": label.title(),
                        "Category": category,
                        "Confidence": f"{conf:.1%}",
                        "Severity": severity,
                        "Box Area (px)": int(box_area)
                    })
            
            # --- DASHBOARD LAYOUT ---
            
            # If no damage detected
            if not detected_objects:
                st.success("✅ No visible exterior damage detected.")
                st.image(image, caption="Original Image", use_container_width=True)
                return

            # If damage detected, show Tabs
            tab1, tab2 = st.tabs(["🔍 Visual Analysis", "📋 Detailed Report"])
            
            with tab1:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Generate Annotated Image
                    annotated_img = results[0].plot() # YOLO's built-in plotter
                    st.image(annotated_img, caption="AI Detection Overlay", use_container_width=True)
                
                with col2:
                    st.subheader("Quick Summary")
                    df = pd.DataFrame(detected_objects)
                    
                    # KPIs
                    total_damages = len(df)
                    high_sev_count = len(df[df['Severity'] == 'High'])
                    
                    st.metric("Total Defects", total_damages)
                    st.metric("Critical / High Severity", high_sev_count, 
                              delta_color="inverse" if high_sev_count > 0 else "normal")
                    
                    st.divider()
                    st.caption("Common Defects Detected:")
                    st.write(df['Type'].value_counts())

            with tab2:
                st.subheader("Assessment Report")
                
                # Color coding for dataframe
                def color_severity(val):
                    color = 'red' if val == 'High' else 'orange' if val == 'Medium' else 'green'
                    return f'color: {color}; font-weight: bold'

                st.dataframe(
                    df.style.map(color_severity, subset=['Severity']), 
                    use_container_width=True,
                    hide_index=True
                )
                
                # Actionable Insight
                st.info("""
                **Recommendation Logic:**
                * **High Severity:** Immediate professional inspection required. Likely structural or glass replacement.
                * **Medium:** Repair shop visit recommended for dent pulling or panel repainting.
                * **Low:** Cosmetic buffing or touch-up paint may suffice.
                """)
                
                # Download Report Button (Simulated)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Inspection CSV",
                    data=csv,
                    file_name='vehicle_damage_report.csv',
                    mime='text/csv',
                )

        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")

if __name__ == "__main__":
    main()