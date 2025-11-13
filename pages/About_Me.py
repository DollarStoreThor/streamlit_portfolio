import streamlit as st

st.title(":red[About Me]")
st.header("Learn more about me!👋",
          divider=True)

col1, col2 = st.columns(2)

with col1.container(horizontal_alignment="distribute"):
    st.markdown("""
    THIS IS A TEST
    Hello! I'm Andrew Thomas, a Machine Learning and Data Scientist with a passion for deep learning and generative AI. 
    With a strong foundation in the Sciences, and hands-on experience in Python programming, I specialize in building innovative AI solutions that drive real-world impact.
    """, 
    width="content")

    st.markdown("""
    I hold two Bachelor's degrees in Microbiology 🦠 and Biochemistry🧪, which have equipped me with a unique perspective on problem-solving and analytical thinking. 
    My journey into the tech world has been fueled by a desire to leverage AI and machine learning to create solutions that can make a difference.
    I'm constantly exploring new technologies and methodologies to stay at the forefront of this rapidly evolving field.
    """)

with col2.container(horizontal_alignment="distribute"):
    st.image("Images/68823420.jpg",
             caption="Andrew Lile Thomas",
             use_container_width=True)

st.markdown("""
My goal is to create impactful solutions, leveraging my unique blend of scientific expertise and technical skills. 
Let’s connect to explore opportunities or discuss exciting tech projects!
""")

outside_links = st.container(horizontal_alignment="center")
with outside_links:
    sub_col1, sub_col2, sub_col3 = st.columns(3, width="stretch")

    sub_col1.markdown("[🔗](https://www.linkedin.com/in/andrew-thomas-5a4305251/)")
    sub_col2.markdown("[🐈‍⬛](https://github.com/DollarStoreThor)")
    sub_col3.markdown("[📧](mailto:andrewlilethomas@gmail.com)")