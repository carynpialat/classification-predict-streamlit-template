"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academ

"""
# Streamlit dependencies
import streamlit as st
import joblib
import os

# Data dependencies
import pandas as pd
import requests
from bs4 import BeautifulSoup
import urllib
import re
from urllib import request
import spacy
nlp = spacy.load('en_core_web_lg')

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl", "rb")
# loading your vectorizer from the pkl file
tweet_cv = joblib.load(news_vectorizer)

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app


def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	#st.title("Climate Change, A hoax or a real threat??")
	#st.subheader("Climate change tweet classification")
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Information", "Prediction"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		#st.info("General Information")
		st.title("Climate Change, A hoax or a real threat??")
		# You can read a markdown file from supporting resources folder
		st.markdown("Climate change is a long-term change in the average weather patterns that have come to define Earth's local, regional and global climates.")
		st.markdown("Changes observed in Earth's climate since the early 20th century are primarily driven by human activities, particularly fossil fuel burning, which increases heat-trapping greenhouse gas levels in Earth's atmosphere, raising Earth's average surface temperature. These human-produced temperature increases are commonly referred to as global warming. Natural processes can also contribute to climate change, including internal variability(e.g., cyclical ocean patterns like Niño, La Niña and the Pacific Decadal Oscillation) and external forcings (e.g., volcanic activity, changes in the Sun's energy output, variations in Earth's orbit.                         https://climate.nasa.gov/resources/global-warming-vs-climate-change/")
		pics = {"Obama": "https://i.insider.com/5ef28394aee6a819e52ef5a7?width=1100&format=jpeg&auto=webp"}
		def person_of_interest(raw):
			trump = []
			data = raw[0:1001]
			#doc=nlp(row['message'])
			for index, row in data.iterrows():
				doc=nlp(row['message'])
				if doc.ents:
					for ent in doc.ents:
						if ent.text == 'Obama':
							trump.append(doc)
			return trump
		capt = person_of_interest(raw)
		st.image(pics["Obama"], use_column_width=True, caption=capt)
		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']])  # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text", "Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(
			    open(os.path.join("resources/Logistic_regression.pkl"), "rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

		if st.button("Quoted article"):
			regex = r'(https?://[^\s]+)'
			url = re.findall(regex, tweet_text)
			for i in url:
				html = request.urlopen(i).read()
				html[:60]
				soup = BeautifulSoup(html, 'html.parser')
				title = soup.find('title')
				return title
			st.success(title)  

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
