from flask import Flask, request, render_template, redirect, url_for
import os
import sqlite3
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def init_db():
    conn = sqlite3.connect('invoices.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS invoices
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, text TEXT)''')
    conn.commit()
    conn.close()

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_features(text):
    invoice_number = re.findall(r'Invoice Number:\s*(\w+)', text)
    date = re.findall(r'Date:\s*(\d{2}/\d{2}/\d{4})', text)
    amount = re.findall(r'Amount:\s*\$?(\d+\.\d{2})', text)
    features = {
        'invoice_number': invoice_number[0] if invoice_number else 'N/A',
        'date': date[0] if date else 'N/A',
        'amount': amount[0] if amount else 'N/A',
        'keywords': set(text.split()),
        'structure': extract_structure(text)
    }
    return features

def extract_structure(text):
    structure = {}
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if 'Invoice Number:' in line:
            structure['invoice_number_line'] = i
        if 'Date:' in line:
            structure['date_line'] = i
        if 'Amount:' in line:
            structure['amount_line'] = i
    return structure

def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

def calculate_structural_similarity(struct1, struct2):
    similarity_score = 0
    total_elements = len(struct1)
    for key in struct1:
        if key in struct2 and struct1[key] == struct2[key]:
            similarity_score += 1
    return similarity_score / total_elements

def find_most_similar_invoice(input_text):
    conn = sqlite3.connect('invoices.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM invoices")
    invoices = cursor.fetchall()
    conn.close()

    most_similar_invoice = None
    highest_combined_similarity_score = 0

    input_features = extract_features(input_text)

    for invoice in invoices:
        invoice_id, filename, text = invoice
        invoice_features = extract_features(text)

        content_similarity_score = calculate_cosine_similarity(input_text, text)
        structural_similarity_score = calculate_structural_similarity(input_features['structure'], invoice_features['structure'])
        combined_similarity_score = (content_similarity_score + structural_similarity_score) / 2

        if combined_similarity_score > highest_combined_similarity_score:
            highest_combined_similarity_score = combined_similarity_score
            most_similar_invoice = {
                'id': invoice_id,
                'filename': filename,
                'similarity_score': combined_similarity_score,
                'invoice_features': invoice_features,
                'content_similarity': content_similarity_score,
                'structural_similarity': structural_similarity_score
            }

    return most_similar_invoice, highest_combined_similarity_score, input_features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        text = extract_text_from_pdf(file_path)

        conn = sqlite3.connect('invoices.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO invoices (filename, text) VALUES (?, ?)", (file.filename, text))
        conn.commit()
        conn.close()

        return redirect(url_for('index'))

@app.route('/compare', methods=['POST'])
def compare_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        input_text = extract_text_from_pdf(file_path)

        most_similar_invoice, combined_similarity_score, input_features = find_most_similar_invoice(input_text)

        return render_template('result.html', 
                               input_features=input_features, 
                               most_similar_invoice=most_similar_invoice)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
