import re
import math
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user,
    login_required, logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Inisialisasi Flask & Config untuk SQLAlchemy ─────────────────────────────
app = Flask(__name__)
app.secret_key = 'your-very-secret-key'  
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root@localhost/jobsearch"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# ─── Inisialisasi SQLAlchemy & Flask-Login ────────────────────────────────────
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.login_view = 'login'   # jika belum login, redirect ke /login
login_manager.init_app(app)

# ─── Model User (Wajib mewarisi UserMixin) ─────────────────────────────────────
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email    = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return f"<User {self.username}>"

# ─── User loader untuk Flask-Login ─────────────────────────────────────────────
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ─── BAGIAN PENCARIAN DENGAN TF-IDF ─────────────────────────────────────────────
df = pd.read_csv('sample_job_descriptions.csv')
df = df[:100000]
df.drop(columns=['Job Id','latitude','longitude','Job Portal','location','Company Profile','Job Description'],
        errors='ignore', inplace=True)

df['Job Title'] = df['Job Title'].fillna('')
if 'skills' in df.columns:
    df['skills'] = df['skills'].fillna('')
elif 'Skills' in df.columns:
    df['skills'] = df['Skills'].fillna('')
else:
    df['skills'] = ''

for col in ['Role','Company','Country','Qualifications','Experience',
            'Salary Range','Work Type','Responsibilities','Benefits','Contact']:
    df[col] = df.get(col, '').fillna('')

def tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.split()

# Gabungkan Title + Role
df['title_role'] = df['Job Title'].fillna('') + ' ' + df['Role'].fillna('')

vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words=None)
tfidf_matrix = vectorizer.fit_transform(df['title_role'])

def search_with_tfidf(query, df, tfidf_matrix, vectorizer):
    query_terms = tokenize(query)
    if not query_terms:
        return []

    query_vec = vectorizer.transform([" ".join(query_terms)])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1]

    feature_names = vectorizer.get_feature_names_out()
    results = []

    for idx in top_indices:
        score = similarities[idx]
        if score <= 0:
            break

        row = df.iloc[idx]
        doc_vec = tfidf_matrix[idx]
        doc_tfidf = dict(zip(
            [feature_names[i] for i in doc_vec.indices],
            doc_vec.data
        ))

        matched_terms = {}
        for term in query_terms:
            if term in doc_tfidf:
                matched_terms[term] = round(float(doc_tfidf[term]), 4)

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
            "Contact": row['Contact'],
            "Score": round(float(score), 4),
            "Matched Terms": matched_terms
        })
    return results

# ─── ROUTES ─────────────────────────────────────────────────────────────────────
@app.route('/')
@login_required
def home():
    return redirect(url_for('search'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('search'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm  = request.form.get('confirm', '').strip()

        if not username or not email or not password or not confirm:
            flash('Please fill out all fields.', 'warning')
            return redirect(url_for('register'))

        if password != confirm:
            flash('Passwords do not match.', 'warning')
            return redirect(url_for('register'))

        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()
        if existing_user:
            flash('Username or email already registered.', 'danger')
            return redirect(url_for('register'))

        hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('search'))

    if request.method == 'POST':
        username_or_email = request.form.get('username_or_email', '').strip()
        password          = request.form.get('password', '').strip()

        user = User.query.filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Logged in successfully.', 'success')
            return redirect(url_for('search'))
        else:
            flash('Invalid credentials. Check username/email & password.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    results     = []
    query       = ""
    page        = 1
    per_page    = 20
    total_pages = 1

    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        try:
            page = int(request.form.get('page', 1))
        except ValueError:
            page = 1

        if query:
            all_results = search_with_tfidf(query, df, tfidf_matrix, vectorizer)
            total_pages = max(1, math.ceil(len(all_results) / per_page))

            if page > total_pages:
                page = total_pages

            start   = (page - 1) * per_page
            end     = start + per_page
            results = all_results[start:end]

    return render_template(
        'search.html',
        results=results,
        query=query,
        page=page,
        total_pages=total_pages
    )

# ─── Buat Tabel Jika Belum Ada & Run App ────────────────────────────────────────
if __name__ == '__main__':
    # with app.app_context():
    #     db.create_all()
    app.run(debug=True)
