# Core Pkgs
import streamlit as st 
import os


# NLP Pkgs
from textblob import TextBlob 
import spacy
# from gensim.summarization import summarize

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Fxn
def convert_to_df(sentiment):
	sentiment_dict = {'polarity':sentiment.polarity,'subjectivity':sentiment.subjectivity}
	sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
	return sentiment_df

def analyze_token_sentiment(docx):
	analyzer = SentimentIntensityAnalyzer()
	pos_list = []
	neg_list = []
	neu_list = []
	for i in docx.split():
		res = analyzer.polarity_scores(i)['compound']
		if res > 0.1:
			pos_list.append(i)
			pos_list.append(res)

		elif res <= -0.1:
			neg_list.append(i)
			neg_list.append(res)
		else:
			neu_list.append(i)

	result = {'positives':pos_list,'negatives':neg_list,'neutral':neu_list}
	return result 
    
# Function for Sumy Summarization
def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result




def main():
	""" NLP Based App with Streamlit """

	# Title
	st.title("SumSent Apps :smiley:")
	st.subheader("Natural Language Processing ")
	st.markdown("""
    	Summarize and Analyze your text
     	""")


	
	# Summarization
   
	if st.form("Show Text Summarization"):
		st.subheader("Summarize Your Text :blue_book: ")
		message = st.text_area("Enter Text")
		summary_options = st.selectbox("Choose Summarizer",['sumy','gensim'])
		if st.button("Summarize"):
			if summary_options == 'sumy':
				st.text("Using Sumy Summarizer ..")
				summary_result = sumy_summarizer(message)
			# elif summary_options == 'gensim':
			# 	st.text("Using Gensim Summarizer ..")
			# 	summary_result = summarize(rawtext)
			# else:
			# 	st.warning("Using Default Summarizer")
			# 	st.text("Using Gensim Summarizer ..")
			# 	summary_result = summarize(rawtext)

		
			st.success(summary_result)

            # Sentiment Analysis
	menu = ["SumSent Apps", "About"]
	choice = st.sidebar.selectbox("Menu",menu)

        
	if choice == "SumSent Apps":
		st.subheader("Want to Check Sentiment For Your Text? :face_with_monocle: ")
        
		with st.form(key='nlpForm'):
			raw_text = st.text_area("Enter Text Here")
			submit_button = st.form_submit_button(label='Analyze')

		# layout
		col1,col2 = st.columns(2)
		if submit_button:

			with col1:
				st.info("Results")
				sentiment = TextBlob(raw_text).sentiment
				st.write(sentiment)

				# Emoji
				if sentiment.polarity > 0:
					st.markdown("Sentiment:: Positive :smiley: ")
				elif sentiment.polarity < 0:
					st.markdown("Sentiment:: Negative :angry: ")
				else:
					st.markdown("Sentiment:: Neutral 😐 ")

				# Dataframe
				result_df = convert_to_df(sentiment)
				st.dataframe(result_df)

				# Visualization
				c = alt.Chart(result_df).mark_bar().encode(
					x='metric',
					y='value',
					color='metric')
				st.altair_chart(c,use_container_width=True)



			with col2:
				st.info("Token Sentiment")

				token_sentiments = analyze_token_sentiment(raw_text)
				st.write(token_sentiments)


	# st.sidebar.subheader("About App")
	# st.sidebar.text("#")
	# st.sidebar.info("#")
	

	# st.sidebar.subheader("By")
	# st.sidebar.text("#")
	# st.sidebar.text("#")
	

if __name__ == '__main__':
	main()
    