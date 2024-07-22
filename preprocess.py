import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Assuming the following columns were used in training
numerical_features = ['email_length', 'subject_length', 'word_count']
text_feature = 'text_without_stopwords'
all_features = numerical_features + [text_feature]

def preprocess_input(email_content):
    # Create a DataFrame from the input email content
    data = {
        'email_length': [len(email_content)],
        'subject_length': [0],  # You can update this if you have subject length
        'word_count': [len(email_content.split())],
        'text_without_stopwords': [email_content]
    }
    
    df = pd.DataFrame(data)
    
    # Define numerical and text transformers
    numerical_transformer_standard = StandardScaler()
    numerical_transformer_minmax = MinMaxScaler()
    text_transformer = TfidfVectorizer(stop_words='english')

    # Fit the transformers (use the saved state if available)
    # For demonstration, fitting here; in production, you should load the fitted transformers
    X_num_standard = numerical_transformer_standard.fit_transform(df[numerical_features])
    X_text = text_transformer.fit_transform(df[text_feature])

    # Combine the numerical and text features
    X_preprocessed = np.hstack((X_num_standard, X_text.toarray()))

    return X_preprocessed
