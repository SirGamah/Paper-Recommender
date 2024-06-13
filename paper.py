# Import packages
import streamlit as st
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data

file_id = '1h3am3IqWft_2RCaQuaM2u6iE-fPa6DTh'  
url = f'https://drive.google.com/uc?id={file_id}'
df = pd.read_csv(url)

df = df.sample(100)


# Initialize the vectorizers
vectorizer_title = TfidfVectorizer(stop_words='english')
vectorizer_abstract = TfidfVectorizer(stop_words='english')

# Fit and transform the titles and abstracts separately
tfidf_matrix_title = vectorizer_title.fit_transform(df['titles'])
tfidf_matrix_abstract = vectorizer_abstract.fit_transform(df['summaries'])

def find_similar_papers(query, vectorizer_title, tfidf_matrix_title, vectorizer_abstract, tfidf_matrix_abstract, top_n=5):
    # Vectorize the user query for titles and abstracts
    query_vec_title = vectorizer_title.transform([query])
    query_vec_abstract = vectorizer_abstract.transform([query])
    
    # Compute cosine similarity between the query and titles
    cosine_sim_title = cosine_similarity(query_vec_title, tfidf_matrix_title).flatten()
    
    # Compute cosine similarity between the query and abstracts
    cosine_sim_abstract = cosine_similarity(query_vec_abstract, tfidf_matrix_abstract).flatten()
    
    # Combine similarities (e.g., average)
    combined_sim = (cosine_sim_title + cosine_sim_abstract) / 2
    
    # Get the indices of the top_n most similar papers
    most_similar_indices = combined_sim.argsort()[-top_n:][::-1]
    
    # Return the details of the most similar papers
    return df.iloc[most_similar_indices]

def highlight_terms(text, terms, color='yellow'):
    # Escape special characters in the terms for regex
    escaped_terms = [re.escape(term) for term in terms.split()]
    # Create a regex pattern to match the terms
    pattern = re.compile(r'(' + '|'.join(escaped_terms) + r')', re.IGNORECASE)
    # Substitute the matched terms with highlighted versions
    highlighted_text = pattern.sub(r'<span style="background-color: {};">\1</span>'.format(color), text)
    return highlighted_text

# Web page setting
page_title = "Research Paper Recommendation App"
page_icon = ":robot"
layout = "centered"

# Page configuration
st.set_page_config(page_title = page_title, page_icon = page_icon, layout = layout)

# Title of the app
st.title('Paper Recommender')

# User input for the query
user_query = st.text_input("Enter your query:")

# Button to get recommendations
if st.button('Find Paper'):
    if user_query:
        results = find_similar_papers(user_query, vectorizer_title, tfidf_matrix_title, vectorizer_abstract, tfidf_matrix_abstract)
        for index, result in results.iterrows():
            highlighted_abstract = highlight_terms(result['summaries'], user_query, color='yellow')
            st.write('**Title:**', result['titles'])
            st.markdown('**Abstract:** ' + highlighted_abstract, unsafe_allow_html=True)
            st.write('----------------')
    else:
        st.write('Please enter a query to search for papers.')


