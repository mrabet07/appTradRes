from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, BartTokenizer, BartForConditionalGeneration
from langdetect import detect, LangDetectException
from concurrent.futures import ThreadPoolExecutor
from flask_caching import Cache
import logging
import re
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})
models = {
    ('en', 'fr'): 'Helsinki-NLP/opus-mt-en-fr',
    ('fr', 'en'): 'Helsinki-NLP/opus-mt-fr-en',
    ('es', 'en'): 'Helsinki-NLP/opus-mt-es-en',
    ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es',
    ('en', 'ar'): 'Helsinki-NLP/opus-mt-en-ar',
    ('ar', 'en'): 'Helsinki-NLP/opus-mt-ar-en',
    ('ar', 'fr'): 'Helsinki-NLP/opus-mt-ar-fr',
    ('fr', 'ar'): 'Helsinki-NLP/opus-mt-fr-ar',
    ('ar', 'es'): 'Helsinki-NLP/opus-mt-ar-es',
    ('es', 'ar'): 'Helsinki-NLP/opus-mt-es-ar',
    ('es', 'fr'): 'Helsinki-NLP/opus-mt-es-fr',
    ('fr', 'es'): 'Helsinki-NLP/opus-mt-fr-es',
}

translation_pipelines = {}
local_model_dir = './models'

# Load Sentence-BERT model
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Load summarization model
summarization_model_name = "facebook/bart-large-cnn"
summarization_tokenizer = BartTokenizer.from_pretrained(summarization_model_name)
summarization_model = BartForConditionalGeneration.from_pretrained(summarization_model_name)

# Load all translation models at startup
for (src_lang, dest_lang), model_name in models.items():
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_model_dir, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=local_model_dir, local_files_only=True)
        translation_pipelines[(src_lang, dest_lang)] = pipeline('translation', model=model, tokenizer=tokenizer)
        logging.debug(f"Loaded model for {src_lang} to {dest_lang}")
    except Exception as e:
        logging.error(f"Error loading model {model_name} for {src_lang} to {dest_lang}: {e}")

executor = ThreadPoolExecutor(max_workers=4)

def calculate_cosine_similarity(reference, text):
    try:
        embeddings = sbert_model.encode([reference, text], convert_to_tensor=True)
        cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return cosine_sim.item() * 100
    except Exception as e:
        logging.error(f"Error calculating cosine similarity: {e}")
        return None

def translate_text(text, src_lang, dest_lang):
    logging.debug(f"Translating from {src_lang} to {dest_lang}: {text}")
    translation_pipeline = translation_pipelines.get((src_lang, dest_lang))
    if translation_pipeline:
        try:
            result = translation_pipeline(text)
            logging.debug(f"Translation result: {result}")
            translation = result[0]['translation_text']
            similarity_score = calculate_cosine_similarity(text, translation)
            return translation, similarity_score
        except Exception as e:
            logging.error(f"Error during translation: {e}")
            return "Translation error. Please try again.", None
    else:
        logging.error(f"Model not found for the specified languages: {src_lang} to {dest_lang}")
        return "Model not found for the specified languages.", None

def summarize_text(text):
    logging.debug(f"Summarizing text: {text}")
    inputs = summarization_tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarization_model.generate(inputs, max_length=500, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    similarity_score = calculate_cosine_similarity(text, summary)
    return summary, similarity_score

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    result_text = ""
    detected_lang = ""
    dest_lang = ""
    similarity_score = None
    action_title = ""
    text_direction = "ltr"  # Default text direction
    
    if request.method == 'POST':
        action = request.form['action']
        dest_lang = request.form.get('dest_lang', '')
        text = request.form['text']

        if text:
            try:
                detected_lang = detect(text)
                logging.debug(f"Detected language: {detected_lang}")
                
                if detected_lang in ['ar', 'he', 'fa', 'ur']:
                    text_direction = "rtl"
                
                if action == 'translate':
                    if (detected_lang, dest_lang) not in models:
                        result_text = "Detected language is not supported for translation."
                        return render_template('dashboard.html', result_text=result_text, detected_lang=detected_lang, dest_lang=dest_lang, text_direction=text_direction, action_title="Translated")
                    
                    cache_key = f"{detected_lang}_{dest_lang}_{text}"
                    cached_translation = cache.get(cache_key)
                    if cached_translation:
                        result_text, similarity_score = cached_translation
                    else:
                        future = executor.submit(translate_text, text, detected_lang, dest_lang)
                        result_text, similarity_score = future.result()
                        cache.set(cache_key, (result_text, similarity_score))
                    action_title = "Translated"
                elif action == 'summarize':
                    result_text, similarity_score = summarize_text(text)
                    action_title = "Summarized"
                else:
                    result_text = "Invalid action selected."
            except LangDetectException:
                result_text = "Could not detect language of the text."

    return render_template('dashboard.html', result_text=result_text, detected_lang=detected_lang, dest_lang=dest_lang, text_direction=text_direction, similarity_score=similarity_score, action_title=action_title)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

class RegisterForm(FlaskForm):
    username = StringField(validators=[
        InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[
        InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError('That username already exists. Please choose a different one.')

    def validate_password(self, password):
        # Custom password validation
        pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).+$'
        if not re.match(pattern, password.data):
            raise ValidationError('Password must contain at least one uppercase letter, one lowercase letter, one digit, and one special character.')

class LoginForm(FlaskForm):
    username = StringField(validators=[
        InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[
        InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Login')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    error_message = None  # Variable to store error message
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            error_message = "Invalid username or password."  # Error message
    return render_template('login.html', form=form, error_message=error_message)

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

if __name__ == "__main__":
    app.run(debug=True)
