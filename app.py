from flask import Flask, request, jsonify, render_template_string
import os
import re
import subprocess
import json
import nltk
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import textract
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.pipeline import make_pipeline

# Initialize Flask app
app = Flask(__name__)

# Initialize NLTK and SpaCy
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Function to extract text from DOC files
def extract_text_from_doc(file_path):
    try:
        result = subprocess.run(['antiword', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            print(f"Failed to extract text from DOC {file_path}: {result.stderr}")
            return None
    except Exception as e:
        print(f"Failed to extract text from DOC {file_path}: {str(e)}")
        return None

# Function to extract text from different file formats
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text

def extract_text(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.txt':
        return extract_text_from_txt(file_path)
    elif file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.doc':
        return extract_text_from_doc(file_path)
    else:
        return textract.process(file_path).decode('utf-8')

def load_files(folder_path):
    file_texts = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            extracted_texts = extract_text(file_path)
            if extracted_texts:
                file_texts[filename] = extracted_texts
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    return file_texts

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    processed_text = ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])
    return processed_text

def vectorize_text(corpus):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    return vectors, vectorizer

def train_random_forest_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_with_model(model, X_test):
    return model.predict(X_test)

def match_job_description(job_description, resumes_vectors, job_descriptions_vectors, job_descriptions_vectorizer, resumes_filenames):
    preprocessed_job_desc = preprocess_text(job_description)
    job_desc_vectorized = job_descriptions_vectorizer.transform([preprocessed_job_desc])
    similarities = cosine_similarity(job_desc_vectorized, resumes_vectors).flatten()
    sorted_indices = similarities.argsort()[::-1]
    ranked_resumes = [(resumes_filenames[idx], similarities[idx]) for idx in sorted_indices]
    results = {
        'Rank': [],
        'Resume': [],
        'Similarity Score': []
    }
    for rank, (resume_filename, similarity) in enumerate(ranked_resumes):
        results['Rank'].append(rank + 1)
        results['Resume'].append(resume_filename)
        results['Similarity Score'].append(similarity)
    results_df = pd.DataFrame(results)
    results_json = results_df.to_json(orient="records", indent=4)
    return results_df, results_json

def filter_resumes_by_percentage(results_df, lower_percent, upper_percent):
    filtered_df = results_df[
        (results_df['Similarity Score'] >= lower_percent / 100) &
        (results_df['Similarity Score'] <= upper_percent / 100)
    ]
    filtered_json = filtered_df.to_json(orient="records", indent=4)
    return filtered_json

# Load and preprocess files
resume_folder = r"D:\Parser\resume_matcher_lambda\Data\Java Developer Resumes"
job_description_folder = r"D:\Parser\resume_matcher_lambda\Data\job_description"
resumes = load_files(resume_folder)
job_descriptions = load_files(job_description_folder)

preprocessed_resumes = {filename: preprocess_text(text) for filename, text in resumes.items()}
preprocessed_job_descriptions = {filename: preprocess_text(text) for filename, text in job_descriptions.items()}

job_descriptions_corpus = list(preprocessed_job_descriptions.values())
job_descriptions_vectors, job_descriptions_vectorizer = vectorize_text(job_descriptions_corpus)
resumes_corpus = list(preprocessed_resumes.values())
resumes_vectors = job_descriptions_vectorizer.transform(resumes_corpus)

job_descriptions_filenames = list(preprocessed_job_descriptions.keys())
resumes_filenames = list(preprocessed_resumes.keys())

# Prepare data for training and testing
corpus = job_descriptions_corpus + resumes_corpus
labels = ['job_description'] * len(job_descriptions_corpus) + ['resume'] * len(resumes_corpus)

X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.2, random_state=42)

# Vectorize text data for training
X_train_vectors, vectorizer = vectorize_text(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train Random Forest Model
rf_model = train_random_forest_model(X_train_vectors, y_train)

# Predict on the test set
predictions = predict_with_model(rf_model, X_test_vectors)
accuracy = (predictions == y_test).mean()

@app.route('/match', methods=['GET'])
def index():
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Job Description Matcher</title>
    </head>
    <body>
        <h1>Job Description Matcher</h1>
        <form id="job-description-form">
            <label for="job_description">Job Description:</label><br>
            <textarea id="job_description" name="job_description" rows="10" cols="50" required></textarea><br><br>
            
            <label for="lower_percent">Lower Percentage:</label><br>
            <input type="number" id="lower_percent" name="lower_percent" min="0" max="100" step="0.1" required><br><br>
            
            <label for="upper_percent">Upper Percentage:</label><br>
            <input type="number" id="upper_percent" name="upper_percent" min="0" max="100" step="0.1" required><br><br>
            
            <input type="button" value="Submit" onclick="submitForm()">
        </form>

        <h2>Results</h2>
        <pre id="result"></pre>

        <script>
            function submitForm() {
                const formData = {
                    job_description: document.getElementById('job_description').value,
                    lower_percent: parseFloat(document.getElementById('lower_percent').value),
                    upper_percent: parseFloat(document.getElementById('upper_percent').value)
                };

                fetch('/match', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('result').textContent = JSON.stringify(data, null, 2);
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            }
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_content)

@app.route('/match', methods=['POST'])
def match():
    data = request.get_json()
    job_description = data.get('job_description')
    lower_percent = data.get('lower_percent')
    upper_percent = data.get('upper_percent')
    
    if not job_description or lower_percent is None or upper_percent is None:
        return jsonify({"error": "Missing parameters"}), 400

    results_df, results_json = match_job_description(job_description, resumes_vectors, job_descriptions_vectors, job_descriptions_vectorizer, resumes_filenames)
    filtered_json = filter_resumes_by_percentage(results_df, lower_percent, upper_percent)
    
    return jsonify(json.loads(filtered_json))

if __name__ == '__main__':
    app.run(debug=True)
