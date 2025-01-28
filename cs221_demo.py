import re
import string
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import torch
from sklearn.metrics import f1_score, log_loss
import streamlit as st

# --- Ánh xạ các từ viết tắt ---
contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have",
                       "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                       "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
                       "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would",
                       "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have",
                       "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
                       "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                       "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
                       "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
                       "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                       "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                       "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",
                       "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
                       "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                       "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is",
                       "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                       "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
                       "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
                       "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                       "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
                       "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will",
                       "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                       "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
                       "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                       "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                       "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are",
                       "you've": "you have", 'u.s': 'america', 'e.g': 'for example'}

# --- Từ điển sửa lỗi chính tả ---
mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
                'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ',
                'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do',
                'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do',
                'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation',
                'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum',
                'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota',
                'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp',
                'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

# --- Hàm clean contractions ---
def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    for word in mapping.keys():
        if "" + word + "" in text:
            text = text.replace("" + word + "", "" + mapping[word] + "")
    # Remove Punctuations
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return text

# --- Hàm correct spelling ---
def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x

# --- Hàm tiền xử lý chính ---
def preprocess_data(text):
    text = clean_contractions(text, contraction_mapping)
    text = correct_spelling(text, mispell_dict)
    return text

# --- Load Models and Tokenizers ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ids = {
    "xlnet": "sercetexam9/cs221-xlnet-large-cased-eng-finetuned-20-epochs-tapt",
    "roberta": "Kuongan/CS221-roberta-base-finetuned-semeval-new",
    "deberta": "sercetexam9/cs221-deberta-base-multi-label-classifier-eng-finetuned-30-epochs-tapt"
}

models = {}
tokenizers = {}

for model_name, model_id in model_ids.items():
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=5)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.to(device)
    model.eval()
    models[model_name] = model
    tokenizers[model_name] = tokenizer
print('SUCCESS!')
# --- Prediction Function (Soft Voting) ---
max_len = 64
emotions = ['anger', 'fear', 'joy', 'sadness', 'surprise']

def predict_emotions_hard_voting(text,threshold=0.5):
    predictions = []
    for model_name, model in models.items():
        tokenizer = tokenizers[model_name]
        inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
            logits = outputs.logits
            probs = torch.sigmoid(logits)  # Use sigmoid for multi-label classification
            preds = (probs >= threshold).int()  # Apply threshold of 0.5 for hard voting
            predictions.append(preds)

    # --- Hard Voting Logic ---
    # Convert list of tensors to a single tensor
    predictions_tensor = torch.cat(predictions, dim=0)
    # Sum predictions across models
    summed_predictions = torch.sum(predictions_tensor, dim=0)
    # Final prediction: 1 if at least 2 models predict 1, otherwise 0
    final_predictions = (summed_predictions >= 2).int()

    return final_predictions.cpu().numpy().flatten()

# --- Streamlit App ---
st.set_page_config(page_title="Sentiment Analysis", page_icon="🔍", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border: none;
        padding: 12px 28px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    .stTextInput>div>div>input {
        padding: 10px;
        font-size: 16px;
        border-radius: 4px;
        border: 1px solid #ccc;
    }
    .stSelectbox>div>div>div {
        padding: 10px;
        font-size: 16px;
        border-radius: 4px;
        border: 1px solid #ccc;
    }
    .stTextArea textarea {
        border: 2px solid #4CAF50;
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }
    h1 {
        text-align: center;
        color: #FF8C00;
        font-family: 'Georgia', serif;
    }
    </style>
    <h1> 🔍 NTA  </h1>
""", unsafe_allow_html=True)

st.title("🔍 Sentiment Analysis Interface")
st.markdown("""
    Welcome to the Sentiment Analysis Interface. Use the form below to enter your sentence and we will predict the emotion of the context.
    """)

text_input = st.text_area("Nhập văn bản vào đây:")

if st.button("Predict", type="primary"):
    start=time.time()
    predictions = predict_emotions_hard_voting(text_input)
    predicted_emotions = [emotions[i] for i, pred in enumerate(predictions) if pred == 1]
    end=time.time()-start
    if predicted_emotions:
        emotion_str = ", ".join(predicted_emotions)
        st.write(f"Cảm xúc dự đoán: {emotion_str}.")
    else:
        st.write("Cảm xúc dự đoán: không thuộc các cảm xúc: anger, fear, joy, sadness hay surprise.")
    st.write(f"Tổng thời gian xử lý: {end}s.")
