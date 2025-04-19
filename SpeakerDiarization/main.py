from audio_to_text import AudioTranscription 
from sentiment_analysis import SentimentAnalysis

def process_audio(audio_file_path):
    transcriber = AudioTranscription()
    result = transcriber.transcribe_audio(audio_file_path)
    df = transcriber.convert_to_df(result)
    
    return df
    # analyzer = SentimentAnalysis()
    # analyzer.add_sentiment_analysis()
    # analyzer.add_sentiment_category()
    # df = analyzer.save_to_csv('user_sentiment.csv')
    

if __name__ == "__main__":
    audio_file_path = "audio_file\AI Advancement.wav"
    process_audio(audio_file_path)
