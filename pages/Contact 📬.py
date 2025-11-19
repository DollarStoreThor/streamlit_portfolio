import streamlit as st
import time


st.set_page_config(page_title="Andrew - Contact",
                   layout="centered",
                   page_icon="📬",
                   initial_sidebar_state="expanded")

st.title(":red[Contact Andrew]")

st.subheader("Get in touch with Andrew! 📬", 
          divider=True)

def typewriter(text, delay=0.04):
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)

def output():
    yield from typewriter("I'm excited to connect with you! Whether you have questions about my projects, want to discuss potential collaborations, or just want to say hello, feel free to reach out.\n\n")
    yield from typewriter("You can contact me through the following channels:\n\n")
    yield from typewriter("- andrewlilethomas@gmail.com\n")
    yield from typewriter("- [LinkedIn](https://www.linkedin.com/in/andrew-thomas-5a4305251/)\n")
    yield from typewriter("- [GitHub](https://github.com/DollarStoreThor)\n")

    yield from typewriter("\nI look forward to hearing from you and exploring how we can connect and collaborate!")

with st.container():
    st.write_stream(i for i in output())