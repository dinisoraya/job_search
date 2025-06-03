from flask import Flask, render_template, request
import pandas as pd
from collections import defaultdict

app = Flask(__name__)

# Load and preprocess dataset
df = pd.read_csv('sample_job_descriptions.csv')
df = df[:100000]
df.drop(columns=['Job Id','latitude', 'longitude','Job Portal', 'location','Company Profile', 'Job Description'], inplace=True)
df['Job Posting Date'] = pd.to_datetime(df['Job Posting Date'])

# Build inverted index
def build_inverted_index(df):
    inverted_index = defaultdict(list)
    for idx, row in df.iterrows():
        terms = str(row['Job Title']).lower().split()
        for term in terms:
            if idx not in inverted_index[term]:
                inverted_index[term].append(idx)
    return inverted_index

inverted_index = build_inverted_index(df)

# Search function
def search_with_inverted_index(query, inverted_index, df, max_results=5):
    query_terms = query.lower().split()
    matching_docs = set()
    for term in query_terms:
        if term in inverted_index:
            matching_docs.update(inverted_index[term])
    matching_docs = list(matching_docs)[:max_results]
    results = []
    
    for doc_id in matching_docs:
        row = df.iloc[doc_id]
        results.append({
            "Job Title": row['Job Title'],
            "Role": row['Role'],
            "Company": row['Company'],
            "Country": row['Country'],
            "Qualifications": row['Qualifications'],
            "Experience": row['Experience'],
            "Salary Range": row['Salary Range'],
            "Work Type": row['Work Type'],
            "Responsibilities": row['Responsibilities'],
            "Skills": row['skills'],
            "Benefits": row['Benefits'],
            "Contact": row['Contact']
        })
    return results

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            results = search_with_inverted_index(query, inverted_index, df)
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
