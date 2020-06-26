import  streamlit as st
import joblib,os
import pandas as pd
import spacy
import requests
from bs4 import BeautifulSoup
import urllib
import re
from urllib import request
import seaborn as sns
import matplotlib.pyplot as plt
nlp = spacy.load('en_core_web_sm')
import wordcloud
from wordcloud import WordCloud

#Vectorizer
tweet_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(tweet_vectorizer)

#Load Raw Data
raw = pd.read_csv("resources/train.csv")

#Load models
def load_model(model_file):
    loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_models

def get_keys(val,my_dict):
    for k,v in my_dict.items():
        if val == v:
            return k

def main():
    """Tweets Classifier """

    options = ["Information", "Exploratory Data Analysis", "Natural Language Processing", "Prediction"]
    selection = st.sidebar.selectbox("Choose Option", options)

    if selection == "Information":
        st.title("Climate Change, A hoax or a real threat??")
        st.markdown("Climate change is a long-term change in the average weather patterns that have come to define Earth's local, regional and global climates.")
        st.markdown("Changes observed in Earth's climate since the early 20th century are primarily driven by human activities, particularly fossil fuel burning, which increases heat-trapping greenhouse gas levels in Earth's atmosphere, raising Earth's average surface temperature. These human-produced temperature increases are commonly referred to as global warming. Natural processes can also contribute to climate change, including internal variability(e.g., cyclical ocean patterns like Niño, La Niña and the Pacific Decadal Oscillation) and external forcings (e.g., volcanic activity, changes in the Sun's energy output, variations in Earth's orbit.                         https://climate.nasa.gov/resources/global-warming-vs-climate-change/")
        st.info("Public views on Climate change")
        pics = {"Obama": "https://i.insider.com/5ef28394aee6a819e52ef5a7?width=1100&format=jpeg&auto=webp"}
        def person_of_interest(raw):
            data = raw[0:700]
            for index, row in data.iterrows():
                doc=nlp(row['message'])
                if doc.ents:
                    for ent in doc.ents:
                        if ent.text == 'Obama':
                            doc
            return doc
        st.image(pics["Obama"], use_column_width=True)
        st.write(person_of_interest(raw))

    #Building out the EDA page
    if selection == "Exploratory Data Analysis":
        st.title("Insight on sampled data")
        st.info("Sentiment Distribution")
        raw.loc[(raw.sentiment == -1),'sentiment']='Anti'
        raw.loc[(raw.sentiment == 0),'sentiment']='Neutral'
        raw.loc[(raw.sentiment == 1),'sentiment']='Pro'
        raw.loc[(raw.sentiment == 2),'sentiment']='News'
        sentiment_count = raw['sentiment'].value_counts()
        sns.barplot(sentiment_count.index, sentiment_count.values, alpha=0.9)
        plt.title('Frequency Distribution of sentiment', fontsize=22)
        plt.ylabel('Count', fontsize=22)
        plt.xlabel('sentiment', fontsize=22)
        st.pyplot()

        st.info("Word frequency")
        #Wordcloud of tweets
        wordcloud2 = WordCloud().generate(' '.join(raw['message']))
        plt.imshow(wordcloud2)
        plt.axis("off")
        st.pyplot()

        st.info("Parts of speech distribution per class")
        def count_pos(raw, sentiment):
            pos_dict = {}
            df_pos = raw[raw['sentiment'] == sentiment]
            for i in range(len(df_pos)):
                text = nlp(raw.iloc[i, 1])
                for j in range(len(text)):
                    part_of_speech = text[j].pos_
                    if part_of_speech in pos_dict.keys():
                        pos_dict[part_of_speech] += 1
                    else:
                        pos_dict[part_of_speech] = 1
            return pos_dict

        if st.checkbox("POS_Anti"):
            grp_neg1_pos = count_pos(raw, 'Anti')
            a = pd.DataFrame.from_dict(grp_neg1_pos, orient='index')
            a.plot(kind='bar')
            st.pyplot()
        if st.checkbox("POS_Neutral"):
            grp_0_pos = count_pos(raw, 'Neutral')
            a = pd.DataFrame.from_dict(grp_0_pos, orient='index')
            a.plot(kind='bar')
            st.pyplot()
        if st.checkbox("POS_Pro"):
            grp_1_pos = count_pos(raw, 'Pro')
            a = pd.DataFrame.from_dict(grp_1_pos, orient='index')
            a.plot(kind='bar')
            st.pyplot()
        if st.checkbox("POS_News"):
            grp_2_pos = count_pos(raw, 'News')
            a = pd.DataFrame.from_dict(grp_2_pos, orient='index')
            a.plot(kind='bar')
            st.pyplot()

        st.info("Entity recognition distribution per class")
        def count_ent(raw, sentiment):
            ent_dict = {}
            name_dict = {}
            df_pos = raw[raw['sentiment'] == sentiment]
            for i in range(len(df_pos)):
                text = nlp(df.iloc[i, 1])
                if text.ents:
                    for ent in text.ents:
                        if ent.label_ in ent_dict.keys():
                            ent_dict[ent.label_] += 1
                        else:
                            ent_dict[ent.label_] = 1
                    for ent in text.ents:
                        if ent.text in name_dict.keys():
                            name_dict[ent.text] += 1
                        else:
                            name_dict[ent.text] = 1
            return ent_dict, name_dict


        if st.checkbox("ENT_Anti"):
            grp_neg1_ent, name_neg1_ent = count_ent(raw, 'Anti')
            b = pd.DataFrame.from_dict(grp_neg1_ent, orient='index')
            b.plot(kind='bar')
            st.pyplot()
        if st.checkbox("ENT_Neutral"):
            grp_0_ent, name_0_ent = count_ent(raw, 'Neutral')
            b = pd.DataFrame.from_dict(grp_0_ent, orient='index')
            b.plot(kind='bar')
            st.pyplot()
        if st.checkbox("ENT_Pro"):
            grp_1_ent, name_1_ent = count_ent(raw, 'Pro')
            b = pd.DataFrame.from_dict(grp_1_ent, orient='index')
            b.plot(kind='bar')
            st.pyplot()
        if st.checkbox("ENT_News"):
            grp_2_ent, name_2_ent = count_ent(raw, 'News')
            a = pd.DataFrame.from_dict(grp_2_pos, orient='index')
            a.plot(kind='bar')
            st.pyplot()
                    
    if selection == "Natural Language Processing":
        st.title("Natural Language Processing")
        tweet = st.text_area("Enter Tweet", "Type Here")
        nlp_task = ["Name_Entity_Recognition","POS_Tags", "URL_extraction"]
        task_choice = st.selectbox("Choose NLP Task",nlp_task)
        if st.button("Analyze"):
            st.info("Original Tweet:: {}".format(tweet))
            docx = nlp(tweet)
            if task_choice == "Name_Entity_Recognition":
                result = [(ent.text,ent.label) for ent in docx.ents]
            elif task_choice == "POS_Tags":
                result = ["'Token':{},'POS':{},'Dependency':{}".format(word.text,word.tag_,word.dep_) for word in docx]
            elif task_choice == "URL_extraction":
                regex = r'(https?://[^\s]+)'
                result = [re.findall(regex, tweet)]
            st.json(result)

    if selection == "Prediction":
        st.title("Text Prediction")
        tweet = st.text_area("Enter Tweet", "Type Here")
        all_ml_models = ["Logistic_Regression","Support_Vector","KNN","Naive Bayes"]
        model_choice = st.selectbox("Choose ML Mode",all_ml_models)

        prediction_labels =  {'Anti':-1,  'Neutral' :0 , 'Pro' :1, 'News':2}
        if st.button("Classify"):
            st.text("Original test ::\n{}".format(tweet))
            vect_text = tweet_cv.transform([tweet]).toarray()
            if model_choice == "Logistic_Regression":
                predictor = load_model("resources/Logistic_regression.pkl")
                prediction = predictor.predict(vect_text)
            elif model_choice == "Support_Vector":
                predictor = load_model("fff.pkl")
                prediction = predictor.predict(vect_text)
            elif model_choice == "KNN":
                predictor = load_model("fff.pkl")
                prediction = predictor.predict(vect_text)
            elif model_choice == "Naive Bayes":
                predictor = load_model("fff.pkl")
                prediction = predictor.predict(vect_text)
            result = get_keys(prediction,prediction_labels)
            st.success("Tweet Categorized as :: {}".format(result))


if __name__ == '__main__':
    main()
