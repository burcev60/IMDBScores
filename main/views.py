from django.shortcuts import render
from django.http import JsonResponse
import joblib
import pickle
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import SnowballStemmer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\\[[^]]*\\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-z\s]'
    text=re.sub(pattern,'',text)
    return text

def remove_stopwords(text):
    tokenizer=ToktokTokenizer()
    stopword_list=nltk.corpus.stopwords.words('english')
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def stem_text(text):
    stemmer = SnowballStemmer('english')
    # Разделяем текст на слова
    words = text.split()
    # Применяем стемминг к каждому слову и объединяем обратно в строку
    return ' '.join([stemmer.stem(word) for word in words])



def home(request):
    if request.method == 'POST':
        review_text = request.POST.get('review').lower()
        review_text = remove_special_characters(denoise_text(review_text))
        review_text = remove_stopwords(review_text)
        review_text = stem_text(review_text) 
        
        vectorizer = joblib.load('main\\static\\tfidf_vectorizer.joblib')
        model  = joblib.load('main\\static\\model.pkl')
             
        processed_text = vectorizer.transform([review_text])         # Предобработка текста

        if not hasattr(vectorizer, 'idf_'):
            raise ValueError("Загруженный векторизатор не был обучен.")

        predicted_score =int(model.predict(processed_text)[0])            # Предсказание с помощью модели
        
        sentiment = 'Positive' if predicted_score > 5 else 'Negative' # Определение окраски
        
        return JsonResponse({
            'sentiment': sentiment,
            'score': predicted_score
        })
    return render(request, 'main/home.html')
