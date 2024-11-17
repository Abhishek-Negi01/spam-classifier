import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email / SMS spam classifier")

input_sms = st.text_area("enter the message :")
if st.button('Predict'):
    #1 preprocess
    def transform_message(text):
        return_list = []
        text = text.lower()
        text = nltk.word_tokenize(text)
        for word in text:
            if word not in string.punctuation and word not in stopwords.words('english'):
                return_list.append(ps.stem(word))
        return " ".join(return_list)

    transformed_sms = transform_message(input_sms)
    #2 vectorize
    vector_input = tfidf.transform([transformed_sms])
    #3 predict
    result = model.predict(vector_input)[0]
    #4 display
    if result == 1:
        st.header('spam')
    else:
        st.header("not spam")

