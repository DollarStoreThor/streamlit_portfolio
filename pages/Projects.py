import streamlit as st


st.title(":red[My Projects]")
st.subheader("Explore Andrew's Projects! 📈", divider=True)


prod1 = st.container(horizontal_alignment="distribute", gap ="medium", border=True)
with prod1:
    col1, col2 = st.columns([1, 1],
                            gap="medium",
                            vertical_alignment="top",)

    with col1.container(horizontal_alignment="distribute"):
        st.markdown("""
        ### SPECTaiCLE: AI-Powered Book Locator 📙
        ***Project Description:***
        Developed an AI-powered book locator that helps users find books in a library using YOLO computer vision.
        The system uses an image to identify book titles and their locations on shelves.
        """)

        st.markdown("""
        ***Role:***\n
            - Developer and Project Lead
        """)

        st.markdown("""
        ***Technologies Used:***\n
            - Python 
            - YOLO
            - Tesseract OCR
            - OpenCV
            - Streamlit
        """)

    with col2.container(horizontal_alignment="distribute", vertical_alignment="center"):
        st.components.v1.iframe("https://www.linkedin.com/embed/feed/update/urn:li:ugcPost:7287009237438803968?compact=1", height=399, width=504, scrolling=True)
        
        st.markdown(" --- ")

        st.page_link("pages/SPECTaiCLE.py", 
                     label="Try SPECTaiCLE for Yourself!", 
                     use_container_width=True,
                     icon="📚")

prod2 = st.container(horizontal_alignment="distribute", gap="medium", border=True)
with prod2:
    col1, col2 = st.columns([1, 1],
                            gap="medium",
                            vertical_alignment="top",)

    with col1.container(horizontal_alignment="distribute"):
        st.markdown("""
        ### Project 2: Data Visualization Dashboard
        Created an interactive data visualization dashboard using Streamlit and Plotly.
        The dashboard allows users to explore complex datasets through dynamic charts and graphs.
        """)

    with col2.container(horizontal_alignment="distribute", vertical_alignment="center"):
        st.image("Images/20251113_140124.jpg", 
                 caption="Project 2: Data Visualization Dashboard", 
                 width="content")


prod3 = st.container(horizontal_alignment="distribute", gap="medium", border=True)
with prod3:
    col1, col2 = st.columns([1, 1],
                            gap="medium",
                            vertical_alignment="top",)

    with col1.container(horizontal_alignment="distribute"):
        st.markdown("""
        ### Project 3: Image Classification Model
        Developed a convolutional neural network for image classification tasks.
        Achieved high accuracy on benchmark datasets using TensorFlow and Keras.
        """)

    with col2.container(horizontal_alignment="distribute", vertical_alignment="center"):
        st.image("Images/20251113_135617.jpg", 
                 caption="Project 3: Image Classification Model", 
                 width="content")

