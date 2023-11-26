from flask import Flask,request, url_for, redirect, render_template
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests 
from bs4 import BeautifulSoup 
import re

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file:
            tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

            model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

            # Process the file using your model
            text = file.read().decode('utf-8')
            tokens = tokenizer.encode(text,return_tensors='pt')
            result1 = model(tokens)
            result = int(torch.argmax(result1.logits))+1
            return render_template('index.html', result=result)
    
    return render_template('index.html', result=None)


if __name__ == '__main__':
    app.run(debug=True)

