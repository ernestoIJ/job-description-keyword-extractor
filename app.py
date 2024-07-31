from flask import Flask, render_template, request, redirect, url_for, session, flash
from text_processing import preprocess_text, extract_keywords
import os

app = Flask(__name__)
secret_key = os.environ.get('SECRET_KEY')
app.secret_key = secret_key

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        job_description = request.form['job_description']
        if 'descriptions' not in session:
            session['descriptions'] = []
        session['descriptions'].append(job_description)
        return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/generate_keywords', methods=['GET'])
def generate_keywords():
    descriptions = session.get('descriptions', [])
    if not descriptions:
        flash("No job descriptions provided.")
        return redirect(url_for('index'))
    preprocess_texts = [preprocess_text(desc) for desc in descriptions]
    keywords = extract_keywords(preprocess_texts, num_keywords=15)
    session.pop('descriptions', None) # Clear the session
    return render_template('results.html', keywords=keywords)

if __name__ == '__main__':
    app.run(debug=True)