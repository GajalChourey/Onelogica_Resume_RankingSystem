Resume Matcher - 
Project Overview
The Resume Matcher application allows users to match resumes with job descriptions based on similarity using natural language processing (NLP) techniques. This Flask-based application leverages TF-IDF vectorization and cosine similarity to rank resumes against a given job description. The app also allows filtering results by similarity percentage ranges.

Main Features:
Text extraction from various file formats including .pdf, .docx, .doc, and .txt.
Preprocessing of text data (removing stopwords, lemmatization, and tokenization).
Matching resumes to job descriptions based on vector similarity using cosine similarity.
Ability to filter resumes based on similarity score percentage ranges.
Displays ranked resumes along with their similarity scores.
Prerequisites

1. Python Libraries
The following Python libraries are used:

Flask: Web framework for creating the web application.
nltk: Natural Language Toolkit for preprocessing text (stopwords, tokenization, lemmatization).
spacy: NLP library for text parsing.
sklearn: Used for TF-IDF vectorization, cosine similarity, and training a RandomForest classifier.
sentence-transformers: Pre-trained BERT model for sentence embeddings.
textract: Library for extracting text from various file formats.
PyPDF2: For extracting text from PDFs.
python-docx: For extracting text from DOCX files.
pandas: For handling and displaying results in dataframes.

2. System Setup
Install all required packages:

How to Run the Project
1. Directory Structure
Create the following folders and place your resume files and job description files in them:
Resumes Folder: Path where resume files (in .pdf, .docx, .txt, .doc formats) are stored.
Job Descriptions Folder: Path where job description files (in text format) are stored.
Modify the resume_folder and job_description_folder variables in the script to point to these directories.

2. Run the Application
Run the Flask app using:

3. Application Workflow
The home page provides a form where the user can:
Paste the job description.
Specify the lower and upper percentage thresholds for filtering resumes based on similarity.
Upon submission, the app matches the resumes with the job description and displays the results in a ranked list format with similarity scores.
4. Testing
The application includes a RandomForest model for classifying resumes and job descriptions for testing purposes. You can also adjust the similarity filters to refine the resume matching process.

Endpoints
1. /match (GET)
Renders the form for submitting the job description and percentage filters.

2. /match (POST)
Accepts the job description, lower and upper similarity percentages, and returns a filtered list of matched resumes in JSON format.

Future Enhancements
Incorporate additional machine learning models for improved accuracy.
Enable file upload for job descriptions and resumes directly via the web interface.
Enhance the filtering mechanism to support additional criteria like experience, skills, etc.
