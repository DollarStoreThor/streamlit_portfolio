import streamlit as st
from PIL import Image
import time

st.set_page_config(page_title="Andrew - About You",
                   layout="centered",
                   page_icon="📍",
                   initial_sidebar_state="expanded")

st.title(":red[About You]")
st.subheader("Learn about where this app is hosted! 📍",
          divider=True)


with st.container():

    st.markdown("Welcome to the About You page! Here, you'll learn about the hardware that powers this web application and how it's hosted.\n")
    st.markdown("This website is hosted on a **Raspberry Pi 4**, a small and affordable computer that you can use to learn programming and build projects.\nRaspberry Pi 4 is equipped with a **Quad-core Cortex-A72 (ARM v8) 64-bit SoC @ 1.5GHz**, making it powerful enough to run various applications, including web apps like this one built with Streamlit.\n")


    st.image("Images/20251113_135617.jpg", caption="Raspberry Pi 4 Hosting the App", use_container_width=True)

    st.markdown("The Raspberry Pi 4 is running a lightweight Linux distribution, optimized for performance and efficiency. This setup allows the web application to run smoothly while consuming minimal resources.\n")
    st.markdown("Hosting on a Raspberry Pi also highlights the potential for low-cost, energy-efficient solutions for web hosting and application deployment. It's a great way to experiment with web technologies without the need for expensive infrastructure or cloud services.\n")

    

    img = Image.open("Images/20251113_140124.jpg")
    # Ensure image is vertical (portrait orientation)
    if img.width > img.height:
        img = img.rotate(-90, expand=True)
    st.image(img, caption="Raspberry Pi 4 Setup", use_container_width=True)

    message = "Thank you for visiting the About You page! I hope you found it informative and inspiring. If you have any questions or would like to learn more about hosting web applications on Raspberry Pi, don't hesitate to get in touch via the:"
    st.markdown(message)
    st.page_link("pages/Contact 📬.py", 
                 label="Contact Page", 
                 use_container_width=False,
                 icon="📬")
    st.markdown("Looking forward to connecting with you!")






