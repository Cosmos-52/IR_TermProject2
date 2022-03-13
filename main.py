import os
from textblob import TextBlob
import string
import pandas as pd
from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import login


def clean(data):
    cleaned_text = data.translate(str.maketrans('', '', string.punctuation + u'\xa0'))
    cleaned_text = cleaned_text.lower().translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), ''))
    tokenized_text = word_tokenize(cleaned_text)
    stop_word = {s: 1 for s in stopwords.words()}
    remove_word = [word for word in tokenized_text if word not in stop_word]
    remove_word = [word for word in remove_word if len(word) > 2]
    ps = PorterStemmer()
    stemmed_word = ' '.join([ps.stem(w) for w in remove_word])
    return stemmed_word


def check_spell(query, query_type):
    check = TextBlob(query).correct()
    correct = str(check)
    print(correct)
    if query_type == 'title':
        query_by_title(correct)
    else:
        query_by_ingredients(correct)


dataset = pd.read_csv('/Users/zhuhongjin/大三下/953481_IR/assignment/IR_TermProject2/asset/archive/recipe.csv')
dataset = dataset.drop(['Unnamed: 0', 'Ingredients'], axis=1)
dataset = dataset.dropna()
dataset = dataset.reset_index(drop=True)
cleaned_title = []
cleaned_ingredients = []

for i in dataset['Title']:
    cleaned_title.append(clean(i))

for i in dataset['Cleaned_Ingredients']:
    cleaned_ingredients.append(clean(i))


def query_by_title(input):
    tfidfvectorizer = TfidfVectorizer(ngram_range=(1, 2))
    title_invert = tfidfvectorizer.fit_transform(cleaned_title)
    query = tfidfvectorizer.transform([clean(input)])
    result = cosine_similarity(title_invert, query).reshape((-1,))
    for i, index in enumerate(result.argsort()[-10:][::-1]):
        print(str(i + 1), dataset['Title'][index], "--", result[index], dataset['Image_Name'][index])


def query_by_ingredients(input):
    tfidfvectorizer = TfidfVectorizer(ngram_range=(1, 2))
    title_invert = tfidfvectorizer.fit_transform(cleaned_ingredients)
    query = tfidfvectorizer.transform([clean(input)])
    result = cosine_similarity(title_invert, query).reshape((-1,))
    for i, index in enumerate(result.argsort()[-10:][::-1]):
        print(str(i + 1), dataset['Title'][index], "--", result[index])


if __name__ == '__main__':
    # login.app.run(host=os.getenv('IP', '0.0.0.0'), port=int(os.getenv('PORT', 3569)))
    check_spell('mushro', 'title')
    check_spell('sugar', 'ingredients')
