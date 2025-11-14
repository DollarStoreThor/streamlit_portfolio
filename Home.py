import streamlit as st

st.set_page_config(page_title="Andrew",
                   layout="centered",
                   page_icon=":briefcase:",
                   initial_sidebar_state="expanded")



col1, col2 = st.columns([1, 1],
                        gap="medium",
                        vertical_alignment="top")

with col1.container(horizontal_alignment="distribute"):
    st.title(":red[Andrew's Portfolio]")
    st.subheader("Welcome to my professional portfolio!👋", divider=True)

    st.markdown("""
    Hello! I'm Andrew Thomas, a Machine Learning and Data Scientist with a passion for deep learning and generative AI. 
    With a strong foundation in the Sciences, and hands-on experience in Python programming, I specialize in building innovative AI solutions that drive real-world impact.
    """, 
    width="content")


    st.markdown("My goal is to create impactful solutions, leveraging my unique blend of scientific expertise and technical skills. " \
    "Let’s connect to explore opportunities or discuss exciting tech projects!")
    
    st.markdown("Feel free to explore my projects, learn more about me, and get in touch!")
    

    sub_col1, sub_col2, sub_col3 = st.columns(3)

    sub_col1.markdown("[LinkedIn](https://www.linkedin.com/in/andrew-thomas-5a4305251/)")
    sub_col2.markdown("[GitHub](https://github.com/DollarStoreThor)")
    sub_col3.markdown("[Email](mailto:andrewlilethomas@gmail.com)")

with col2.container(horizontal_alignment="distribute", ):
    st.image("Images/Andrew_Fullbody.png") # Replace with your image URL
    
with st.container(horizontal_alignment="center", border=True):
        st.text("Data Scientist  |  Machine Learning Engineer  |  Tech Enthusiast  | Problem Solver")

with st.sidebar:
    # Sidebar content
    print("Sidebar loaded")