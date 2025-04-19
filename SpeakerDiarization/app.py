import streamlit as st
import pandas as pd
import os
from PIL import Image, UnidentifiedImageError
import seaborn as sns
import matplotlib.pyplot as plt
from main import process_audio
from feedback_generation import ConversationReviewer
from sentiment_analysis import SentimentAnalysis


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("uploaded_files", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return os.path.join("uploaded_files", uploaded_file.name)
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None



def get_conversation(df):
    conversation = ""
    for index, row in df.iterrows():
        conversation += f"{row['speaker']}: {row['text']} "
    return conversation

st.title("Speech To Text Analysis")



# Display the image below the title
img_path = "image.png"
if os.path.exists(img_path):
    try:
        # Try opening the image file with PIL to check for validity
        image = Image.open(img_path)
        st.image(image, caption="Speech To Text Analysis", use_column_width=True)
    except UnidentifiedImageError:
        st.error("The image file could not be opened. It might be in an unsupported or corrupted format.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
else:
    st.error(f"Image not found at: {img_path}")


    
st.sidebar.title("Options")
option = st.sidebar.selectbox("Choose an option", ("Upload an audio file", "Use default audio"))




def display_data(df):
    # Display the dataframe with specified columns
    st.write("### Speech to Text Results with Speaker Identification")
    st.dataframe(df[['start', 'end', 'text', 'speaker']])


    # Analyze Sentiment
    sentiment_review = SentimentAnalysis()
    sentiment = get_conversation(df)
    review_sentiment = sentiment_review.review_sentiment(sentiment)
    # df_result = sentiment_review.parse_output(review_sentiment)
    
    ## For Plotting 
    # st.write("### Sentiment Category by Speaker")
    # fig, ax = plt.subplots(figsize=(10, 5))
    # sns.countplot(data=df_result, x='speaker', ax=ax)
    # st.pyplot(fig)

    st.write("### Sentiment Review")
    st.write(review_sentiment)




    # Analyze conversation
    conversation_reviewer = ConversationReviewer()
    conversation = get_conversation(df)
    review_output = conversation_reviewer.review_conversation(conversation)

    st.write("### Conversation Review")
    st.write(review_output)




## UPLOADING THE FILE
if option == "Upload an audio file":
    uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=["wav", "mp3"])
    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        if file_path:
            st.sidebar.success("File uploaded successfully!")
            df = process_audio(file_path)
            display_data(df)

elif option == "Use default audio":
    default_audio_path = "audio_file/AI Advancement.wav"
    df = process_audio(default_audio_path)
    display_data(df)
