

import os
import pandas as pd

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class SentimentAnalysis:
    def __init__(self):
        load_dotenv()
        groq_api_key = os.getenv("WHISPER_GROQ")
        self.llm = ChatGroq(temperature=0.5, groq_api_key=groq_api_key, model_name="llama3-8b-8192")
        self.prompt_template = """
        You are Sentiment Analyzer. Your Job is to analyze the sentiment of the person. 
        Generate the Sentiments of all unique speakers after analyzing their speech.
        Also give a remark if the sentiment is positive, negative or netural.
        conversation: {conversation}
        """
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["conversation"]
        )
        self.output_chain = self.prompt | self.llm | StrOutputParser()

    def review_sentiment(self, conversation):
        search_output = self.output_chain.invoke({"conversation": conversation})
        return search_output




## NOT USED NOW
    def parse_output(self, sentiment_output):
        """
        Parses the sentiment output from the model, extracting speakers and sentiments.

        :param sentiment_output: Output from the sentiment analysis model (string format)
        :return: DataFrame with columns 'speaker' and 'sentiment'
        """
        parsed_data = []
        
        # Split the output into lines for processing
        lines = sentiment_output.splitlines()

        for line in lines:
            # Try to handle cases where the output format is not as expected
            if ":" in line:
                try:
                    speaker, sentiment = line.split(":")
                    speaker = speaker.strip()
                    sentiment = sentiment.strip()

                    # Add speaker and sentiment data to the list
                    if speaker and sentiment:
                        parsed_data.append({"speaker": speaker, "sentiment": sentiment})
                except ValueError:
                    print(f"Skipping line due to parsing error: {line}")
            else:
                print(f"Skipping line as no ':' separator found: {line}")

        # If parsed_data is empty, return None to avoid further errors
        if not parsed_data:
            return None
        
        # Convert to pandas DataFrame
        return pd.DataFrame(parsed_data)

## NOT USED NOW
    def save_to_csv(self, dataframe, csv_filename="sentiment_output.csv"):
        dataframe.to_csv(csv_filename, index=False)
        print(f"Sentiment data saved to {csv_filename}")

