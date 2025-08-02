import os
import streamlit as st
import numpy as np
import cv2


st.markdown(
    """
    <style>
    body {
        background-color: #6E6454;
    }
    .stApp {
        background-color: #bca66c;
    }
    .stSidebar {
        background-color: #6E6454;
    }
    .img:hover::after {
            content: attr(alt);
        }
    </style>
    """,
    unsafe_allow_html=True
)



st.set_page_config(layout="wide",
                    # page_title="EcliPixel",
                    initial_sidebar_state="collapsed")

# ================= Header =================
col1, col2 = st.columns([1, 9])
with col1:
    st.image("assets/logo.png", width=100) 
    # st.markdown("### EcliPixel ")


# ================= Layout =================
left_col, right_col = st.columns([1.2, 2])

with left_col:


    square_size = 700
    image = np.zeros((square_size, square_size, 3), dtype=np.uint8)
    image[:, :] = [69, 69, 69]

    # Load the overlay image (ensure it has alpha channel)
    overlay_path = "assets/Add Image.png"
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    
    if overlay is not None:
        # Resize overlay if larger than square
        max_overlay_size = int(square_size * 0.5)
        h, w = overlay.shape[:2]
        scale = min(max_overlay_size / h, max_overlay_size / w, 1.0)
        new_size = (int(w * scale), int(h * scale))
        overlay_resized = cv2.resize(overlay, new_size, interpolation=cv2.INTER_AREA)

        # Calculate position to center overlay
        y_offset = (square_size - overlay_resized.shape[0]) // 2
        x_offset = (square_size - overlay_resized.shape[1]) // 2

        # Overlay with alpha channel
        if overlay_resized.shape[2] == 4:
            alpha = overlay_resized[:, :, 3] / 255.0
            for c in range(3):
                image[y_offset:y_offset+overlay_resized.shape[0], x_offset:x_offset+overlay_resized.shape[1], c] = (
                    alpha * overlay_resized[:, :, c] +
                    (1 - alpha) * image[y_offset:y_offset+overlay_resized.shape[0], x_offset:x_offset+overlay_resized.shape[1], c]
                ).astype(np.uint8)
        else:
            image[y_offset:y_offset+overlay_resized.shape[0], x_offset:x_offset+overlay_resized.shape[1], :] = overlay_resized[:, :, :3]

    st.image(image, caption="")

    # Para customizar a largura do botão, use st.markdown com HTML/CSS:
    st.markdown(
        """
        <style>
        .custom-btn {
            width: 700px; /* Altere para a largura desejada, ex: 300px */
            height: 50px;
            font-size: 18px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        </style>
        <button class="custom-btn">Add Image</button>
        """,
        unsafe_allow_html=True
    )

#########################################################################
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(
            f"""
            <button style="border:none;background:none;padding:0;">
            <img src="assets/Group 17.png" width="100" style="width:100%;max-width:100px;" />
            </button>
            """,
            unsafe_allow_html=True
        )


