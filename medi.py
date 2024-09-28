import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
# Load dataset
data = pd.read_csv('medicine.csv')

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Data cleaning function
def clean_text(text):
    # Remove non-alphabetic characters
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Apply the cleaning function to the 'Description' column
data['cleaned_description'] = data['Description'].apply(clean_text)
