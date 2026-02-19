import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="AutoInspect AI - Intelligent Damage Assessment",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
[data-testid="stMetric"] {
    background-color: #ffffff !important;
    padding: 16px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}
[data-testid="stMetric"] label {
    color: #000000 !important;
    font-weight: 600;
}
[data-testid="stMetric"] div {
    color: #000000 !important;
    font-size: 26px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

def calculate_severity(box_area, image_area):
    ratio = box_area / image_area
    if ratio < 0.02:
        return "Low"
    elif ratio < 0.10:
        return "Medium"
    else:
        return "High"

def categorize_damage_type(label):
    functional_keywords = ['glass', 'broken', 'crack', 'lamp']
    if any(k in label.lower() for k in functional_keywords):
        return "Functional (Critical)"
    return "Cosmetic (Surface)"

@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(path)

def main():
    with st.sidebar:
        st.title("⚙️ Inspection Settings")
        st.info("Upload a clear vehicle image")
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)

    st.title("🚗 AutoInspect AI")
    st.markdown("### Intelligent Vehicle Damage Detection & Assessment System")

    model = load_model("best.pt")

    uploaded_file = st.file_uploader(
        "Drop vehicle image here...",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is None:
        st.info("👆 Upload an image to begin inspection")
        return

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    h, w, _ = img_array.shape
    image_area = h * w

    with st.spinner("🔍 Analyzing vehicle surface..."):
        results = model.predict(
            img_array,
            conf=conf_threshold,
            device="cpu",
            verbose=False
        )

    detected_objects = []

    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_area = (x2 - x1) * (y2 - y1)
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            detected_objects.append({
                "Type": label.title(),
                "Category": categorize_damage_type(label),
                "Confidence": f"{conf:.1%}",
                "Severity": calculate_severity(box_area, image_area),
                "Box Area (px)": int(box_area)
            })

    if not detected_objects:
        st.success("✅ No visible exterior damage detected")
        st.image(image, use_container_width=True)
        return

    df = pd.DataFrame(detected_objects)

    tab1, tab2 = st.tabs(["🔍 Visual Analysis", "📋 Detailed Report"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            annotated = results[0].plot()
            annotated = Image.fromarray(annotated)
            st.image(annotated, caption="AI Detection Overlay", use_container_width=True)

        with col2:
            st.subheader("Quick Summary")
            st.metric("Total Defects", len(df))
            st.metric("Critical / High Severity", len(df[df["Severity"] == "High"]))
            st.divider()
            st.caption("Common Defects Detected")
            st.write(df["Type"].value_counts())

    with tab2:
        st.subheader("Assessment Report")

        def color_severity(val):
            if val == "High":
                return "color:red;font-weight:bold"
            elif val == "Medium":
                return "color:orange;font-weight:bold"
            else:
                return "color:green;font-weight:bold"

        st.dataframe(
            df.style.map(color_severity, subset=["Severity"]),
            use_container_width=True,
            hide_index=True
        )

        st.info("""
        **Repair Recommendation Logic**
        - **High:** Immediate professional inspection
        - **Medium:** Repair shop visit recommended
        - **Low:** Cosmetic fix sufficient
        """)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Inspection CSV",
            csv,
            "vehicle_damage_report.csv",
            "text/csv"
        )

if __name__ == "__main__":
    main()
