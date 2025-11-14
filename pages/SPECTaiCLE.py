import streamlit as st

st.title(":red[SPECT]ai:red[CLE]")
st.subheader("Explore the SPECTaiCLE Project! 📚", divider=True)

image = st.file_uploader("Upload Image of your Bookshelf", 
                         type=["png", "jpg", "jpeg"])
if image is not None:
    st.image(image, caption="Uploaded Bookshelf Image", 
             use_container_width=True)

with st.container():
    message = st.chat_input("Ask me about the books on your shelf!", key="chat_input")
    if message:
        st.chat_message("user").markdown(message)

        with st.chat_message("assistant"):
            st.markdown("Analyzing your bookshelf image and retrieving book information...")
            # Here you would add the code to process the image and respond to the user's query.
            # For demonstration purposes, we'll use a placeholder response.
            st.markdown("Here are some books I found on your shelf:\n\n1. Book Title 1 by Author A\n2. Book Title 2 by Author B\n3. Book Title 3 by Author C")

st.button("Clear Chat", on_click=lambda: (st.session_state.update({"messages": []})))
