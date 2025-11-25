
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


# lower case
def transform_Message(Message):
    Message = Message.lower()
    # tokenisation
    Message = nltk.word_tokenize(Message)

    # now text is a list, so we run a loop
    y = []

    for i in Message:
        if i.isalnum():  # if i is alphanumeric it gets appended
            y.append(i)
    # removal of stopwords and punctuations
    Message = y[:]  # Use y here, which contains alphanumeric words
    y.clear()
    for i in Message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            # lemmatization
    text = y[:]  # Make a copy of y before clearing it for stemming
    y.clear()
    for i in text:  # Use text, not Message here
        y.append(ps.stem(i))

    return " ".join(y)


tfidf =pickle.load(open('../PythonProject5/vectorizer.pkl','rb'))
model=pickle.load(open('../PythonProject5/model.pkl','rb'))



st.title("Email spam classifier")
input_message=st.text_input("Enter the message")




if st.button('predict'):

    # preprocess
    transformed_message = transform_Message(input_message)

    # vectorise

    vector_input = tfidf.transform([transformed_message])

    # predict
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam Message")
    else:
        st.header("Valid Message")


