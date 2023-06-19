import streamlit as st
import hashlib
import pandas as pd
import pickle
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
import os
# Set the password for accessing the app
PASSWORD_HASH = "c0a16a726686f7c44f99536443e6b942ba4cd80e5bd81a739ab63698a4368302"

# Load the trained model and vectorizer
classifier = pickle.load(open(r'finalized_model5.pkl', 'rb'))
vectorizer = pickle.load(open(r'vectorizer.pkl', 'rb'))

# Define the Streamlit app
def main():
    # Check if the user is authorized to access the app
    if not check_credentials():
        return

    st.title('Model Prediction')
    st.write('Upload a CSV file to get predictions')
    
    def get_keywords():
        keywords = []
        st.write('Enter keywords (one per line):')
        keyword_input = st.text_area('Keywords')
        if keyword_input:
            keywords = keyword_input.strip().split('\n')
        return keywords

    # Define your predefined keywords
    keywords = get_keywords()
    
    # Create a file uploader
    file = st.file_uploader('Upload CSV file', type=['csv'])

    # Perform predictions when a file is uploaded
    if file is not None:
        df = pd.read_csv(file, encoding='latin-1')
        df = df.astype(str)
        
        # Load your DataFrame with research papers
        # Preprocess the text by combining the title and abstract
        df['Text'] = df['Title'] + ' ' + df['Abstract']

        # Vectorize the text using TF-IDF
        tfidf_matrix = vectorizer.transform(df['Text'])
        df['Relevance'] = 0
        
        # Calculate the similarity between the keywords and each paper
        keyword_vector = vectorizer.transform(keywords)
        similarity_scores = cosine_similarity(keyword_vector, tfidf_matrix)
        df1 = pd.DataFrame(similarity_scores.max(axis=0), columns=['Sim'])
        df1 = df1.astype('float')
        
        for index, row in df.iterrows():
            title = row['Title']
            abstract = row['Abstract']
    
            # Calculate relevance score based on keywords
            relevance = sum(keyword in title.lower() or keyword in abstract.lower() for keyword in keywords)
    
            # Update the relevance score in the DataFrame
            df.at[index, 'Relevance'] = relevance
            
        # Define the weights for relevance and similarity
        relevance_weight = 15.0
        similarity_weight = 20.0

        # Create the feature matrix X based on keyword relevance and similarity
        tfidf_matrix = vectorizer.transform(df['Text'])
        X_sim = csr_matrix(df1['Sim'].values.reshape(-1, 1))
        X_sim_repeated = hstack([X_sim] * tfidf_matrix.shape[1])

        # Multiply the relevance and similarity features by the weights
        relevance_features = df['Relevance'].values.reshape(-1, 1) * relevance_weight
        similarity_features = X_sim_repeated * similarity_weight

        # Combine the features with the weighted relevance and similarity
        features = hstack([tfidf_matrix, relevance_features, similarity_features])
        X = features
                                    
        # Perform predictions using the loaded model and vectorizer
        # Replace the following code with your prediction logic
        predictions = classifier.predict(X)
        
        # Add the predictions to the DataFrame
        df['Prediction'] = predictions
        
        # Display the DataFrame with predictions
        st.write('Predictions:', df)
        
        # Add a button to save the DataFrame to a CSV file
        if st.button('Save Predictions'):
            save_to_csv(df)

def save_to_csv(df):
    # Save the DataFrame to a CSV file in the "Downloads" directory
    st.download_button(
        label="Download Predictions CSV",
        data=df.to_csv().encode('utf-8'),
        file_name='predictions.csv',
        mime='text/csv'
    )

def check_credentials():
    password = st.sidebar.text_input("Enter password", value="", type="password")
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    if password_hash != PASSWORD_HASH:
        st.sidebar.error("Invalid password. Access denied.")
        return False
    return True

if __name__ == '__main__':
    main()
