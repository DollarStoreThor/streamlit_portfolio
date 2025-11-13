import streamlit as st

st.set_page_config(page_title="Andrew",
                   layout="centered",
                   page_icon=":briefcase:",
                   initial_sidebar_state="expanded")


st.title(":red[Andrew's Portfolio] \n ### Welcome to my professional portfolio! 👋")

col1, col2 = st.columns([1, 1],
                        gap="medium",
                        vertical_alignment="top",)
with col1.container(horizontal_alignment="distribute"):
    st.markdown("Hello! I'm Andrew Thomas, an aspiring Machine Learning and Data Scientist with a passion for deep learning and generative AI. " \
    "With a strong foundation in Biochemistry and hands-on experience in Python programming, I specialize in building innovative AI solutions that drive real-world impact.",
    width="stretch")


    st.markdown("My goal is to create impactful solutions, leveraging my unique blend of scientific expertise and technical skills. " \
    "Let’s connect to explore opportunities or discuss exciting tech projects!")
    
    st.markdown("Feel free to explore my projects, learn more about me, and get in touch!")
    

    sub_col1, sub_col2, sub_col3 = st.columns(3)

    sub_col1.markdown("[LinkedIn](https://www.linkedin.com/in/andrew-thomas-5a4305251/)")
    sub_col2.markdown("[GitHub](https://github.com/DollarStoreThor)")
    sub_col3.markdown("[Email](mailto:andrewlilethomas@gmail.com)")

with col2.container(horizontal_alignment="distribute"):
    st.image("https://avatars.githubusercontent.com/u/68823420?s=400&u=b433ebbb41252c2072708bdd806d75fccfde07dd&v=4",) # Replace with your image URL
    st.markdown("##### Data Scientist | Machine Learning Engineer | Tech Enthusiast")


with st.sidebar:
    # Sidebar content
    print("Sidebar loaded")