# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:48:00 2024

@author: rites
"""

import pickle
import streamlit as st 
import numpy as np

st.header('Books recommendation system using clusteing technique')
model=pickle.load(open('model.pkl','rb'))
book_name=pickle.load(open('book_name.pkl','rb'))
final_rating=pickle.load(open('final_rating.pkl','rb'))
book_pivot=pickle.load(open('book_pivot.pkl','rb'))

def fecth_poster(suggestion):
    book_name=[]
    ids_index=[]
    poster_url=[]
    
    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])
    
    for name in book_name[0]:
        ids=np.where(final_rating['Book-Title'] ==name)[0][0]
        ids_index.append(ids)
        
    for idx in ids_index:
        url=final_rating.iloc[idx]['Image-URL-L']
        poster_url.append(url)
        
    return poster_url    

def recommend_books(book_name):
    book_list=[]
    book_id=np.where(book_pivot.index == book_name)[0][0]
    distance,suggestion=model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1),n_neighbors=6)
    
    poster_url=fecth_poster(suggestion)
    
    for i in range(len(suggestion)):
        books= book_pivot.index[suggestion[i]]
        for j in books:
            book_list.append(j)
            
    return book_list,poster_url

selected_books=st.selectbox(
    'Type or select a book',
    book_name
    )

if st.button('Show Recommendation'):
    recommend_books,poster_url=recommend_books(selected_books)
    col1, col2, col3, col4, col5=st.columns(5)
    
    with col1:
        st.text(recommend_books[1])
        st.image(poster_url[1])
        
    with col2:
        st.text(recommend_books[2])
        st.image(poster_url[2])
        
    with col3:
        st.text(recommend_books[3])
        st.image(poster_url[3])
        
    with col4:
        st.text(recommend_books[4])
        st.image(poster_url[4])
        
    with col5:
        st.text(recommend_books[5])
        st.image(poster_url[5])
