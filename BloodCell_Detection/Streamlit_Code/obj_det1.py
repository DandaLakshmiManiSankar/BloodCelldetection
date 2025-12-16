import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import os
os.environ["ULTRALYTICS_NO_CV2"] = "1"

# Load trained YOLOv10 model
MODEL_PATH = "/content/runs/detect/train/weights/best.pt"  # Update with actual path
model = YOLO(MODEL_PATH)

st.title("Blood Cell Detection using YOLOv10")
st.write("Upload an image to detect blood cells.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert image to tensor and run YOLO inference
    results = model(image)
    
    # Draw bounding boxes
    draw = ImageDraw.Draw(image)
    detection_data = []
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = r.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), f"{label} {confidence:.2f}", fill="red")
            detection_data.append([label, confidence, x1, y1, x2, y2])
    
    # Show output image
    st.image(image, caption="Detected Objects", use_column_width=True)
    
    # Show detection results in a table
    if detection_data:
        st.write("### Detection Results")
        df = pd.DataFrame(detection_data, columns=["Class", "Confidence", "X1", "Y1", "X2", "Y2"])
        st.dataframe(df)
    
    # Display precision and recall for each class and overall
    #st.write("### Precision and Recall")
   # metrics = model.val()  # Evaluate model performance
    #precision_list = metrics.box.p.tolist()
    #ecall_list = metrics.box.r.tolist()
    
    #precision_recall_data = []
    
    #for class_id, class_name in model.names.items():
     #   precision = precision_list[class_id] if class_id < len(precision_list) else 0
      #  recall = recall_list[class_id] if class_id < len(recall_list) else 0
       # precision_recall_data.append([class_name, precision, recall])
    
    # Overall precision and recall
   # overall_precision = metrics.box.mp  # Mean precision
    #overall_recall = metrics.box.mr  # Mean recall
    #precision_recall_data.append(["Overall", overall_precision, overall_recall])
    
    #df_metrics = pd.DataFrame(precision_recall_data, columns=["Class", "Precision", "Recall"])
    #st.dataframe(df_metrics)

