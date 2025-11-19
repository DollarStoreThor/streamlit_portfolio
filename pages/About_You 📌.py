import streamlit as st
from PIL import Image

st.set_page_config(page_title="Andrew - About You",
                   layout="centered",
                   page_icon="📍",
                   initial_sidebar_state="expanded")

st.title(":red[About You]")
st.subheader("Learn about where this app is hosted! 📍",
          divider=True)

st.markdown("""
This website is hosted on a **Raspberry Pi 4**, a small and affordable computer that you can use to learn programming and build projects.
The Raspberry Pi 4 is equipped with a **Quad-core Cortex-A72 (ARM v8) 64-bit SoC @ 1.5GHz**, making it powerful enough to run various applications, including web apps like this one built with Streamlit.
""")

st.image("Images/20251113_135617.jpg", caption="Raspberry Pi 4 Hosting the App", use_container_width=True)

st.markdown("""
By hosting this app on a Raspberry Pi 4, I aim to demonstrate the capabilities of edge computing and showcase how even compact devices can handle modern web applications and services.
Feel free to explore the app and see how it performs on this versatile hardware!
""")

img = Image.open("Images/20251113_140124.jpg")
# Ensure image is vertical (portrait orientation)
if img.width > img.height:
    img = img.rotate(-90, expand=True)
st.image(img, caption="Raspberry Pi 4 Setup", use_container_width=True)

st.markdown("If you're interested in learning more about Raspberry Pi or how to host your own apps, feel free to reach out!")