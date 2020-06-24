import  streamlit as st
import joblib,os
import pandas as pd
import spacy

nlp = spacy.load('en_core_web_lg')
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
    st.title("Tweet Classifier")
    st.subheader("Climate change tweet classification")

    activities = ["NLP","Prediction","Information"]
    choice = st.sidebar.selectbox("Choose Activity", activities)

    if choice == "NLP":
        st.info("Natural Language Processing")
        tweet = st.text_area("Enter Tweet", "Type Here")
        nlp_task = ["Tokenization","Name_Entity_Recognition","Lemmatization","POS_Tags"]
        task_choice = st.selectbox("Choose NLP Task",nlp_task)
        if st.button("Analyze"):
            st.info("Original Tweet:: {}".format(tweet))
            docx = nlp(tweet)
            if task_choice == "Tokenization":
                result= [token.text for token in docx]
                st.json(result)
            elif task_choice == "Name_Entity_Recognition":
                    result = [(entity.text,entity.label) for entity in docx.ents]

            elif task_choice == "Lemmatization":
                    result = ["'Token':{},'lemma_':{}".format(token.text,token.lemma_) for token in docx]
            elif task_choice == "POS_Tags":
                    result = ["'Token':{},'POS':{},'Dependency':{}".format(word.text,word.tag_,word.dep_) for word in docx]
            st.json(result)
        if st.button("NLP_Table"):
            docx = nlp(tweet)
            c_tokens = [token.text for token in docx]
            c_lemma = [token.lemma_ for token in docx]
            c_pos = [word.tag_ for word in docx]

            new_df = pd.DataFrame(zip(c_tokens,c_lemma,c_pos),columns =['Tokens','Lemma','POS'])
            st.dataframe(new_df)







    if choice == "Prediction":
        st.info("Prediction with ML")

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


    if choice == "Information":
        st.info("Information")






if __name__ == '__main__':
    main()
