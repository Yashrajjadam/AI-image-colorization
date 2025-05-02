import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import torch
import RRDBNet_arch as arch
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="AI Image Processor", layout="wide")

def set_container_style():
    st.markdown(
        """
        <style>
        /* Main page background */
        .stApp {
            background-color: #e6f4ff;
            color: #003366;
            font-family: 'Arial', sans-serif;
        }
        h1 {
            color: #0059b3;
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        .description {
            color: #003366;
            text-align: center;
            margin-bottom: 2rem;
            font-size: 1.2rem;
        }
        .upload-section, .output-section {
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0px 4px 6px rgba(0, 85, 170, 0.2);
            padding: 20px;
            margin: 20px;
        }
        .upload-section {
            border: 2px dashed #66b3ff;
        }
        .button-container {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        button[kind="primary"] {
            background-color: #66b3ff !important;
            border: none;
            border-radius: 8px;
            color: white !important;
            font-size: 1rem;
            padding: 10px 20px;
        }
        .button-container button:hover {
            background-color: #3399ff !important;
        }
        /* Image display */
        .stImage {
            border-radius: 8px;
            overflow: hidden;
        }
        /* Footer styling */
        .footer {
            margin-top: 40px;
            text-align: center;
            font-size: 0.9rem;
            color: #003366;
        }
        .footer a {
            color: #0059b3;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_colorization_model(model_path="E:/Minor 1/ZIP/color_image_checkpoint.keras"):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading colorization model: {e}")
        return None

@st.cache_resource
def load_super_resolution_model(model_path="E:/Minor 1/ZIP/RRDB_ESRGAN_x4.pth"):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()
        model.to(device)
        return model, device
    except Exception as e:
        st.error(f"Error loading super-resolution model: {e}")
        return None, None

def enhance_image(image, model, device):
    try:
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(np.transpose(image[:, :, [2, 1, 0]], (2, 0, 1))).float()
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(image_tensor).data.squeeze().float().cpu().clamp_(0, 1).numpy()

        output_image = np.transpose(output_tensor[[2, 1, 0], :, :], (1, 2, 0))
        return (output_image * 255.0).round().astype(np.uint8)
    except Exception as e:
        st.error(f"Error enhancing image: {e}")
        return image

def get_image_download_link(image, format="PNG"):
    try:
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        buffer = BytesIO()
        image_pil.save(buffer, format=format)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error creating download link: {e}")
        return None

def daltonize_image(image):
    try:
     
        correction_matrix = np.array(
            [[0.56667, 0.43333, 0], 
             [0.55833, 0.44167, 0], 
             [0, 0.24167, 0.75833]]
        )
        daltonized_image = np.dot(image / 255.0, correction_matrix.T)
        daltonized_image = np.clip(daltonized_image * 255.0, 0, 255).astype(np.uint8)
        return daltonized_image
    except Exception as e:
        st.error(f"Error creating Daltonized image: {e}")
        return image

def main():
    set_container_style()
    st.title("📸 AI Image Processor")
    st.markdown(
        "<p class='description'>Upload your image, and let our advanced AI colorize, enhance, and simulate Daltonization!</p>",
        unsafe_allow_html=True,
    )

    color_model = load_colorization_model()
    sr_model, device = load_super_resolution_model()

    if color_model is None or sr_model is None:
        st.error("Models couldn't be loaded. Please try again later.")
        st.stop()

    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        process_button = st.button("🛠️ Process Image")

        if process_button:
            try:
                with st.spinner("Colorizing the image..."):
                    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    gray_image = cv2.resize(gray_image, (160, 160)).astype(np.float32) / 255.0
                    gray_image = np.expand_dims(np.expand_dims(gray_image, axis=-1), axis=0)
                    colorized_image = color_model.predict(gray_image)[0]
                    colorized_image = (np.clip(colorized_image, 0, 1) * 255).astype(np.uint8)

                with st.spinner("Enhancing the colorized image..."):
                    enhanced_image = enhance_image(colorized_image, sr_model, device)

                if image.shape[0] > 160 or image.shape[1] > 160:
                    final_image = cv2.resize(enhanced_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
                else:
                    final_image = enhanced_image

                with st.spinner("Generating Daltonized image..."):
                    daltonized_image = daltonize_image(final_image)

                st.markdown("<div class='output-section'>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.image(image, caption="Original Image", use_container_width=True)

                with col2:
                    st.image(final_image, caption="Processed Image", use_container_width=True)

                with col3:
                    st.image(daltonized_image, caption="Daltonized Image", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                final_image=cv2.cvtColor(final_image,cv2.COLOR_RGB2BGR)
                buffer = get_image_download_link(final_image, format="PNG")
                st.download_button(
                    label="📥 Download Processed Image",
                    data=buffer,
                    file_name="processed_image.png",
                    mime="image/png",
                )

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

    st.markdown(
        "<div class='footer'>"
        "Made with ❤️ using Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
