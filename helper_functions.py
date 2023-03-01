import numpy as np
import pandas as pd
import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import snscrape.modules.twitter as sntwitter
import datetime as dt
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
nltk.download(
    ["punkt", "wordnet", "omw-1.4", "averaged_perceptron_tagger", "universal_tagset", "stopwords"]
)
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import plotly.io as pio
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image


# Create a custom plotly theme and set it as default
pio.templates["custom"] = pio.templates["plotly_white"]
pio.templates["custom"].layout.margin = {"b": 25, "l": 25, "r": 25, "t": 50}
pio.templates["custom"].layout.width = 600
pio.templates["custom"].layout.height = 450
pio.templates["custom"].layout.autosize = False
pio.templates["custom"].layout.font.update(
    {"family": "Arial", "size": 12, "color": "#707070"}
)
pio.templates["custom"].layout.title.update(
    {
        "xref": "container",
        "yref": "container",
        "x": 0.5,
        "yanchor": "top",
        "font_size": 16,
        "y": 0.95,
        "font_color": "#353535",
    }
)
pio.templates["custom"].layout.xaxis.update(
    {"showline": True, "linecolor": "lightgray", "title_font_size": 14}
)
pio.templates["custom"].layout.yaxis.update(
    {"showline": True, "linecolor": "lightgray", "title_font_size": 14}
)
pio.templates["custom"].layout.colorway = [
    "#1F77B4",
    "#FF7F0E",
    "#54A24B",
    "#D62728",
    "#C355FA",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#FFE323",
    "#17BECF",
]
pio.templates.default = "custom"

current_time = datetime.datetime.now()
date = current_time.strftime("%Y-%m-%d")

def get_latest_tweet_df(search_term, num_tweets):
    tweet_data = []
    for i, tweet in enumerate(
        sntwitter.TwitterSearchScraper("{} lang:id".format(search_term)).get_items()
    ):
        if i >= num_tweets or i >= 1000:
            break
        tweet_data.append(
            [tweet.user.username, tweet.date, tweet.likeCount, tweet.content]
        )
    tweet_dataset = pd.DataFrame(
       tweet_data, columns=["Username", "Date", "Like Count", "Tweet"]
    )
    tweet_df = tweet_dataset.drop_duplicates(subset=['Tweet'], keep='last')
    return tweet_df


# membuat fungsi untuk case folding
def casefolding(tweet_df):
    pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    pattern1 = re.compile(r'pic.twitter.com/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    tweet_df = re.sub(pattern,' ',tweet_df) #remove urls if any
    tweet_df = re.sub(pattern1,' ',tweet_df)
    #Convert to lower case
    tweet_df = tweet_df.lower()                                
     #Convert www.* or https?://*
    tweet_df = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet_df) 
    #Convert @username to AT_USER
    tweet_df = re.sub('@+','',tweet_df)
     #Remove additional white spaces
    tweet_df = re.sub('[\s]+', ' ', tweet_df)
    #Replace #word with word
    tweet_df  = re.sub(r'#([^\s]+)', r'\1', tweet_df )
    #remove symbols
    tweet_df = re.sub(r'[^.,a-zA-Z0-9 \n\.]',' ',tweet_df)
    tweet_df = tweet_df.replace(',',' ').replace('.',' ')          
    tweet_df = re.sub(r'[^\w\s]', '', tweet_df)    
    #trim             
    tweet_df = tweet_df.strip('\'"')
    return tweet_df

key_norm = pd.read_csv('https://raw.githubusercontent.com/FaddliLWibowo/Twitter-Sentiment-Analysis-with-Streamlit/main/Dataset/kamus_kbba.csv', encoding='ISO-8859-1')

def text_normalize(tweet_df):
    tweet_df = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0]
    if (key_norm['singkat'] == word).any()
    else word for word in tweet_df.split()
    ])
    tweet_df = str.lower(tweet_df)
    return tweet_df

stopwords_ind = stopwords.words('indonesian')
more_stopword = ['terusmajubersamaprabowo', 'mendingprabowo', 'prabowopresiden', 'prabowopersatuanindonesia', 'prabowosubianto']
stopwords_ind = stopwords_ind + more_stopword

def remove_stop_word(tweet_df):
    clean_words = []
    tweet_df = tweet_df.split()
    for word in tweet_df:
        if word not in stopwords_ind:
            clean_words.append(word)
    return " ".join(clean_words)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# membuat fungsi untuk stemming bahasa indonesia
def stemming(tweet_df):
    tweet_df = stemmer.stem(tweet_df)
    return tweet_df

# membuat fungsi untuk menggabungkan seluruh langkah text preprocessing
def text_preprocessing(tweet_df):
    tweet_df = casefolding(tweet_df)
    tweet_df = text_normalize(tweet_df)
    tweet_df = remove_stop_word(tweet_df)
    tweet_df = stemming(tweet_df)
    return tweet_df

def predict_sentiment(tweet_df):
    model = load_model("static/twitter-sentiment-analysis-model-lstm.h5")
    with open("static/tokenizer.pickle", "rb") as handle:
        custom_tokenizer = pickle.load(handle)
    temp_df = tweet_df.copy()
    temp_df["Cleaned Tweet"] = temp_df["Tweet"].apply(text_preprocessing)
    temp_df = temp_df[(temp_df["Cleaned Tweet"].notna()) & (temp_df["Cleaned Tweet"] != "")]
    sequences = pad_sequences(
        custom_tokenizer.texts_to_sequences(temp_df["Cleaned Tweet"]), maxlen=54
    )
    score = model.predict(sequences)
    temp_df["Score"] = score
    temp_df["Sentiment"] = temp_df["Score"].apply(
        lambda x: "Positif" if x >= 0.50 else "Negatif"
    )
    return temp_df


def plot_sentiment(tweet_df):
    sentiment_count = tweet_df["Sentiment"].value_counts()
    fig = px.pie(
        values=sentiment_count.values,
        names=sentiment_count.index,
        hole=0.3,
        title="<b>Sentiment Distribution</b>",
        color=sentiment_count.index,
        color_discrete_map={"Positive": "#1F77B4", "Negative": "#FF7F0E"},
    )
    fig.update_traces(
        textposition="inside",
        texttemplate="%{label}<br>%{value} (%{percent})",
        hovertemplate="<b>%{label}</b><br>Percentage=%{percent}<br>Count=%{value}",
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_wordcloud(tweet_df, colormap="Greens"):
    stopwords = set()
    with open("Dataset/stopwords.txt", "r") as file:
        for word in file:
            stopwords.add(word.rstrip("\n"))
    cmap = mpl.cm.get_cmap(colormap)(np.linspace(0, 1, 20))
    cmap = mpl.colors.ListedColormap(cmap[10:15])
    mask = np.array(Image.open("Static/twitter_mask.png"))
    font = "static/quartzo.ttf"
    text = " ".join(tweet_df["Cleaned Tweet"])
    wc = WordCloud(
        background_color="white",
        font_path=font,
        stopwords=stopwords,
        max_words=90,
        colormap=cmap,
        mask=mask,
        random_state=42,
        collocations=False,
        min_word_length=2,
        max_font_size=200,
    )
    wc.generate(text)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    # plt.title("Wordcloud", fontdict={"fontsize": 16}, fontweight="heavy", pad=20, y=1.0)
    return fig


def get_top_n_gram(tweet_df, ngram_range, n=10):
    stopwords = set()
    with open("Dataset/stopwords.txt", "r") as file:
        for word in file:
            stopwords.add(word.rstrip("\n"))
    corpus = tweet_df["Cleaned Tweet"]
    vectorizer = CountVectorizer(
        analyzer="word", ngram_range=ngram_range, stop_words=stopwords
    )
    X = vectorizer.fit_transform(corpus.astype(str).values)
    words = vectorizer.get_feature_names_out()
    words_count = np.ravel(X.sum(axis=0))
    df = pd.DataFrame(zip(words, words_count))
    df.columns = ["words", "counts"]
    df = df.sort_values(by="counts", ascending=False).head(n)
    df["words"] = df["words"].str.title()
    return df


def plot_n_gram(n_gram_df, title, color="#54A24B"):
    fig = px.bar(
        x=n_gram_df.counts,
        y=n_gram_df.words,
        title="<b>{}</b>".format(title),
        text_auto=True,
    )
    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(title=None)
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Count=%{x}", marker_color=color)
    return fig