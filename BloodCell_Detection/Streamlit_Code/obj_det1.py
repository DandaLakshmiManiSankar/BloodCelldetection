import os
os.environ["ULTRALYTICS_NO_CV2"] = "1"

import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import pandas as pd

MODEL_PATH = "best.pt"   # model must be in same folder
model = YOLO(MODEL_PATH)

st.title("Blood Cell Detection using YOLOv10")
st.write("Upload an image to detect blood cells.")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model(image)

    draw = ImageDraw.Draw(image)
    detection_data = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = r.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), f"{label} {confidence:.2f}", fill="red")

            detection_data.append(
                [label, confidence, x1, y1, x2, y2]
            )

    st.image(image, caption="Detected Blood Cells", use_column_width=True)

    if detection_data:
        df = pd.DataFrame(
            detection_data,
            columns=["Class", "Confidence", "X1", "Y1", "X2", "Y2"]
        )
        st.write("### Detection Results")
        st.dataframe(df)
