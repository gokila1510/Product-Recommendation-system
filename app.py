import streamlit as st 
import pickle as pkl
import numpy as np
import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

st.header("""Code Crusaders""")
st.title("Product recommendation")

product = st.selectbox("Select the product you need to buy ",["Flax Seeds - Roasted","Lemon & Tea Tree Oil Soap","Face Wash - Oil Control, Active","Dove Plastic Soap Case - Assorted Colour","Sesame Seed Oil","Sugar Free Petit Beurre - The Taste of France","Dog Supplement - Absolute Skin + Coat Tablet","Extra Fine Green Peas","Pet Solitaire Container Set - Silver","Fruit Power - Masala Sugarcane","Veggie Cutter","Choco Deck - Mini Delights","Dhania - Dal","Sport Deo Spray - Fresh, for Men","Aloevera Litchi Ice Tea","Soap - Saffron"])
#st.sidebar.write("Model")
#st.sidebar.write("Tfidf Vectorizer")

st.write("Selected keyword: \n", product)
st.write("Model: Tfidf Vectorizer")

#dataset
uploaded_file = st.file_uploader("BigBasket.csv", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
df.drop('index', axis=1, inplace=True)
df.drop_duplicates(inplace = True, subset=['product'])

st.button("submit")
#model
df = df.dropna()
df = df.reset_index(drop=True)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
mapping = pd.Series(df.index,index=df['product'])


def recommend_product_based_on_click(product_input):
    product_index = mapping[product_input]
    #get similarity values with other product
    #similarity_score is the list of index and similarity matrix
    similarity_score = list(enumerate(cosine_sim[product_index]))
    #sort in descending order the similarity score of product inputted with all the other product
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    # Get the scores of the 20 most similar products. Ignore the first product.
    similarity_score = similarity_score[1:20]
    #return product names using the mapping series
    product_indices = [i[0] for i in similarity_score]
    return (df['product'].iloc[product_indices])



st.title("Recommended Products")
st.write(recommend_product_based_on_click(product))