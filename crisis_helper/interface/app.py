import streamlit as st
import datetime
import requests

st.title("Crisis Helper 🚨")
import streamlit as st

# You can also apply CSS styling to the title
st.markdown(
    """
    <style>
        /* Add your custom CSS styling here */
        .streamlit-title {
            font-size: 36px;
            color: #FF5733; /* Change to your desired title color */
        }
    </style>
    """,
    unsafe_allow_html=True
)

with st.form(key='params_for_api'):

    user_text = st.text_input('What is your tweet?','')
    user_image = st.file_uploader('Upload an image', type=["jpg", "png", "jpeg"])

    st.form_submit_button('Upload')

if st.button("Show me the result"):
    if not user_text and not user_image:
        st.warning("Please enter text or upload an image.")
    else:
        # Prepare the data to send to the API
        data = {}
        if user_text:
            data["text"] = user_text

        if user_image:
            # Read and encode the image
            image = Image.open(user_image)
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="JPEG")
            image_bytes = image_bytes.getvalue()

            # Add the image data to the request
            files = {"image": ("image.jpg", image_bytes)}

params = dict(user_text=user_text, user_image=user_image)

api_url = 'https:?????'
response = requests.post(api_url, data=data, files=files if user_image else None)

prediction = response.json()

pred = prediction['label']

# Display the API response
if response.status_code == 200:
            result = response.json()
            st.success("Prediction Result:")
            st.json(result)
else:
            st.error("Prediction failed. Please try again later.")
