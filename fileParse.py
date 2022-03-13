import numpy as np
import pandas as pd
import string
from textblob import TextBlob
import nltk

import fileParse

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv(
    '/Users/zhuhongjin/大三下/953481_IR/assignment/TermProject2/asset/archive/Food Ingredients and Recipe Dataset with Image Name Mapping.csv')
# title = data.pop('Title')

# remove the row that the content is broken
tmp = data['Title']
nan = tmp.isnull()
for i, v in nan.items():
    if v:
        data = data.drop(index=i)
        # print(i)


# fix = data['Title']
# test = fix.isnull()
# for i,v in test.items():
#     if(v):
#         print(i)

# dropedData = data.drop(columns=data.columns[:1], axis=1).drop(['Cleaned_Ingredients'], axis=1).drop(['Instructions'], axis=1)
# title = dropedData['Title'].squeeze()
def check_spell(query, query_type):
    if query is '':
        return print('please fill something')
    check = TextBlob(query).correct()
    correct = str(check)
    if query_type == 'title':
        search_by_title([correct])
    else:
        search_by_ingredients([correct])



def get_cleandata_and_unique(data):
    clean_data = data.apply(lambda s: s.translate(str.maketrans('', '', string.punctuation + u'\xa0')))
    clean_data = clean_data \
        .apply(lambda s: s.lower()) \
        .apply(lambda s: s.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), ''))) \
        .drop_duplicates()
    clean_data = clean_data.apply(lambda s: [x.strip() for x in s.split()])

    sw_set = set(stopwords.words())
    no_sw_description = clean_data.apply(lambda x: [w for w in x if w not in sw_set])
    ps = PorterStemmer()
    stemmed_description = no_sw_description.apply(lambda x: set([ps.stem(w) for w in x]))
    all_unique_term = list(set.union(*stemmed_description.to_list()))
    invert_idx = {}
    for s in all_unique_term:
        invert_idx[s] = set(stemmed_description.loc[stemmed_description.apply(lambda x: s in x)].index)
    return clean_data, invert_idx


def search_by_title(query):
    invert_idx = fileParse.invert_title
    to_search_title = query  # query should be list
    ps = PorterStemmer()
    stemmed_title = np.unique([ps.stem(w) for w in to_search_title])
    searched_title = sorted(set.union(*[invert_idx[s] for s in stemmed_title]))
    print(query, searched_title)
    return searched_title


def search_by_ingredients(query):
    invert_idx = fileParse.invert_ingredients
    to_search_ingredients = query  # query should be list
    ps = PorterStemmer()
    stemmed_ingredients = np.unique([ps.stem(w) for w in to_search_ingredients])
    searched_ingredients = sorted(set.union(*[invert_idx[s] for s in stemmed_ingredients]))
    print(query, searched_ingredients)
    return searched_ingredients


title = data['Title']
tmpTitle = get_cleandata_and_unique(title)
clean_title, invert_title = tmpTitle

ingredients = data['Ingredients']
temIngredients = get_cleandata_and_unique(ingredients)
clean_ingredients, invert_ingredients = temIngredients

if __name__ == '__main__':
    check_spell('mushro', 'title')
    check_spell('sugar', 'ingredients')

# -------------------
# clean_title = title.apply(lambda s: s.translate(str.maketrans('', '', string.punctuation + u'\xa0')))
# clean_title = clean_title \
#     .apply(lambda s: s.lower()) \
#     .apply(lambda s: s.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), ''))) \
#     .drop_duplicates()
# clean_title = clean_title.apply(lambda s: [x.strip() for x in s.split()])
#
# sw_set = set(stopwords.words())
# no_sw_description = clean_title.apply(lambda x: [w for w in x if w not in sw_set])
# ps = PorterStemmer()
# stemmed_description = no_sw_description.apply(lambda x: set([ps.stem(w) for w in x]))
# all_unique_term = list(set.union(*stemmed_description.to_list()))
#
# invert_idx = {}
# for s in all_unique_term:
#     invert_idx[s] = set(stemmed_description.loc[stemmed_description.apply(lambda x: s in x)].index)
#
# to_search_title = ['Peanut']
# stemmed_title = np.unique([ps.stem(w) for w in to_search_title])
# searched_title = sorted(set.union(*[invert_idx[s] for s in stemmed_title]))
#
# print(clean_title.loc[searched_title].apply(lambda x: ' '.join(x)).head().to_markdown())
# ---------------------------
