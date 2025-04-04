import streamlit as st
st.set_page_config(
    layout="wide",
    page_title="NEWSIFY",
    page_icon="📰"
)

# Core Libraries
import requests
from datetime import datetime, timedelta, timezone
import os
import re
import time
import hashlib # For creating Firestore document IDs from URLs
import traceback # For detailed error logging

# Data Handling & ML
import pandas as pd # Keep pandas for potential future data handling needs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Firebase
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, auth, firestore, exceptions

# Email & Notifications
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl

# Streamlit Components
from streamlit.components.v1 import html

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
# NewsAPI
API_KEY = os.getenv("NEWSAPI_KEY") or st.secrets.get("NEWSAPI_KEY")
BASE_URL = "https://newsapi.org/v2/everything"
HEADLINES_URL = "https://newsapi.org/v2/top-headlines"
DEFAULT_PAGE_SIZE = 30 # Articles per page for standard feeds
NEWSAPI_TOP_HEADLINE_CATEGORIES = {'business', 'entertainment', 'general', 'health', 'science', 'sports', 'technology'}

# Firebase Credentials (from .env or Streamlit secrets)
FIREBASE_CONFIG = {
    "type": os.getenv("FIREBASE_TYPE") or st.secrets.get("FIREBASE_TYPE"),
    "project_id": os.getenv("FIREBASE_PROJECT_ID") or st.secrets.get("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID") or st.secrets.get("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": (os.getenv("FIREBASE_PRIVATE_KEY") or st.secrets.get("FIREBASE_PRIVATE_KEY","")).replace('\\n', '\n'), # Handle newline characters
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL") or st.secrets.get("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID") or st.secrets.get("FIREBASE_CLIENT_ID"),
    "auth_uri": os.getenv("FIREBASE_AUTH_URI") or st.secrets.get("FIREBASE_AUTH_URI"),
    "token_uri": os.getenv("FIREBASE_TOKEN_URI") or st.secrets.get("FIREBASE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_CERT_URL") or st.secrets.get("FIREBASE_AUTH_PROVIDER_CERT_URL"),
    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL") or st.secrets.get("FIREBASE_CLIENT_CERT_URL")
}

# Email Configuration (optional, for notifications)
SMTP_SERVER = os.getenv("SMTP_SERVER") or st.secrets.get("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT") or st.secrets.get("SMTP_PORT", 587))
SMTP_USERNAME = os.getenv("SMTP_USERNAME") or st.secrets.get("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD") or st.secrets.get("SMTP_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM") or st.secrets.get("EMAIL_FROM")
APP_NAME = "Personalized News Aggregator"
EMAIL_CONFIGURED = all([SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, EMAIL_FROM])

# Other Settings
PLACEHOLDER_IMAGE = "https://via.placeholder.com/400x200?text=No+Image+Available"

# --- Initial Application Checks ---
if not API_KEY:
    st.error("Fatal Error: NewsAPI Key (NEWSAPI_KEY) is missing. Please configure it in your .env file or Streamlit secrets.")
    st.stop()
if not all(FIREBASE_CONFIG.values()) or not FIREBASE_CONFIG["private_key"]:
    st.error("Fatal Error: Firebase configuration is missing or incomplete. Please check your .env file or Streamlit secrets (ensure FIREBASE_PRIVATE_KEY is correctly formatted).")
    st.stop()
if not EMAIL_CONFIGURED and not os.getenv("IS_LOCAL_DEV"): # Only warn prominently if not explicitly local dev
     print("Warning: Email configuration incomplete. Login/Signup notifications will be disabled.")

# --- Firebase Initialization ---
# Use session state to ensure Firebase is initialized only once per session
if 'firebase_initialized' not in st.session_state:
    st.session_state.firebase_initialized = False
if not st.session_state.firebase_initialized:
    try:
        # Check if Firebase app is already initialized in this process/thread
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_CONFIG)
            firebase_admin.initialize_app(cred)
            st.session_state.firebase_initialized = True
            print("Firebase Initialized Successfully.")
        else:
            st.session_state.firebase_initialized = True # App already exists
            print("Firebase App already initialized.")
    except ValueError as e: # Catch specific error for invalid credentials format
        st.error(f"Fatal Error: Failed to initialize Firebase due to invalid credentials format. Check FIREBASE_PRIVATE_KEY. Error: {e}")
        traceback.print_exc()
        st.session_state.firebase_initialized = False
        st.stop()
    except Exception as e: # Catch other initialization errors
        st.error(f"Fatal Error: Failed to initialize Firebase: {str(e)}")
        traceback.print_exc()
        st.session_state.firebase_initialized = False
        st.stop()

# Verify Firestore connection after potential initialization
if st.session_state.get("firebase_initialized", False):
    try:
        db = firestore.client()
        # Perform a simple read operation to test connection and permissions
        db.collection('_test_connection').document('_test').get(timeout=5) # 5 second timeout
        print("Firestore connection test successful.")
    except exceptions.PermissionDeniedError:
        st.error("Fatal Error: Firestore permissions error. The service account might lack necessary read/write permissions. Check Firestore Security Rules and IAM roles.")
        st.stop()
    except exceptions.FailedPreconditionError as e:
        st.error(f"Fatal Error: Firestore Query Error: {str(e)}. Ensure Firestore is enabled in your Firebase project (Native mode).")
        st.stop()
    except Exception as e: # Catch other potential errors like network timeout
        st.error(f"Fatal Error: Failed to verify Firestore connection: {str(e)}")
        traceback.print_exc()
        st.stop()
else:
     # This state should ideally not be reached if init failed, but acts as a safeguard
     if not st.session_state.get("firebase_init_error_shown"):
        st.error("Fatal Error: Firebase could not be initialized.")
        st.session_state.firebase_init_error_shown = True
     st.stop()

# --- Categories Definition ---
# (Structure with display names, emojis, colors, and subcategory search terms)
CATEGORIES = {
    '_personalized_': {
        'display_name': '✨ Personalized', 'emoji': '✨', 'color': '#6f42c1',
        'subcategories': {'For You': '_for_you_', 'Saved Articles': '_saved_', 'Liked Articles': '_liked_'}
    },
    'technology': {
        'display_name': 'Technology', 'emoji': '💻', 'color': '#007bff',
        'subcategories': {
            'General Tech': 'technology', 'AI & ML': '"artificial intelligence" OR "machine learning"',
            'Gadgets': 'gadgets OR consumer electronics', 'Software Dev': '"software development" OR programming',
            'Cybersecurity': 'cybersecurity OR hacking', 'Startups & VC': 'startup OR venture capital'
        }
    },
    'business': {
        'display_name': 'Business', 'emoji': '💼', 'color': '#28a745',
        'subcategories': {
            'General Biz': 'business', 'Markets': 'stock market OR finance', 'Economy': 'economy OR inflation',
            'Corporate': 'corporate earnings OR mergers', 'Personal Fin': '"personal finance" OR budget'
        }
    },
    'science': {
        'display_name': 'Science', 'emoji': '🔬', 'color': '#17a2b8',
        'subcategories': {
            'General Sci': 'science', 'Space': 'space OR astronomy OR NASA', 'Environment': 'environment OR climate change',
            'Physics': 'physics OR quantum', 'Biology': 'biology OR genetics'
        }
    },
     'health': {
        'display_name': 'Health', 'emoji': '❤️', 'color': '#dc3545',
        'subcategories': {
            'General Health': 'health', 'Medicine': 'medical research OR disease', 'Wellness': 'wellness OR fitness OR nutrition',
            'Policy': '"healthcare policy" OR FDA'
        }
    },
    'sports': {
        'display_name': 'Sports', 'emoji': '⚽', 'color': '#ffc107',
        'subcategories': {
            'General Sports': 'sports', 'Football (Soccer)': 'football OR soccer', 'Basketball': 'basketball OR NBA',
            'Am. Football': '"american football" OR NFL', 'Tennis': 'tennis', 'Olympics': 'olympics'
        }
    },
    'entertainment': {
        'display_name': 'Entertainment', 'emoji': '🎬', 'color': '#fd7e14',
        'subcategories': {
            'General Ent.': 'entertainment', 'Movies': 'movie OR film', 'Music': 'music OR concert',
            'TV': 'television OR "TV show"', 'Gaming': 'video game OR gaming', 'Celebrity': 'celebrity'
        }
    },
    'general': {
        'display_name': 'General', 'emoji': '📰', 'color': '#6c757d',
        'subcategories': {
            'World News': 'world news', 'US News': '"US news"', 'Politics': 'politics', 'Culture': 'culture OR arts'
        }
    }
}

# --- Recommendation System Setup ---
# Initialize TF-IDF Vectorizer globally for reuse
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000, # Limit vocabulary size for performance/memory
    min_df=2,          # Ignore terms that appear in less than 2 documents
    max_df=0.85        # Ignore terms that appear in more than 85% of documents (too common)
)

# --- Email Notification Functions ---
def send_email(to_email, subject, body, is_html=False):
    """Sends email via configured SMTP server."""
    if not EMAIL_CONFIGURED:
        print("Error: send_email called but email is not configured.")
        return False
    try:
        msg = MIMEMultipart()
        msg['From'] = f"{APP_NAME} <{EMAIL_FROM}>" # Set display name
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html' if is_html else 'plain', 'utf-8')) # Specify encoding
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server: # Use timeout
            server.ehlo()
            server.starttls(context=context) # Secure connection
            server.ehlo()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        print(f"Email sent successfully to {to_email} with subject '{subject}'.")
        return True
    except Exception as e:
        # Show error in UI *and* log detailed error
        st.error(f"Error sending email: {str(e)}")
        print(f"Error sending email to {to_email}:")
        traceback.print_exc()
        return False

def send_login_notification(email):
    """Sends notification email after successful login."""
    if not EMAIL_CONFIGURED:
        print("Login email skipped: Not configured.")
        return False
    subject = f"Successful Login to {APP_NAME}"
    body = f"""
    <html><body>
    <p>Hello,</p>
    <p>You successfully logged in to your {APP_NAME} account at {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}.</p>
    <p>If this was not you, please change your password or contact support immediately.</p>
    <p>Thank you,<br/>The {APP_NAME} Team</p>
    </body></html>
    """
    return send_email(email, subject, body, is_html=True)

def send_signup_notification(email):
    """Sends welcome email after successful signup."""
    if not EMAIL_CONFIGURED:
        print("Welcome email skipped: Not configured.")
        return False
    subject = f"Welcome to {APP_NAME}!"
    body = f"""
    <html><body>
    <p>Hello,</p>
    <p>Thank you for creating an account with {APP_NAME}! We're excited to have you.</p>
    <p>Start exploring news categories, discover personalized content in the 'For You' section (it learns from your activity!), and save your favorite articles.</p>
    <p>Happy reading!<br/>The {APP_NAME} Team</p>
    </body></html>
    """
    return send_email(email, subject, body, is_html=True)

# --- Helper Functions ---
def format_date(date_str):
    """Formats ISO date string or Firestore Timestamp nicely."""
    if isinstance(date_str, datetime): # Handle Firestore Timestamps
        dt_obj = date_str.replace(tzinfo=timezone.utc) if date_str.tzinfo is None else date_str
        try: return dt_obj.astimezone().strftime("%b %d, %Y %I:%M %p %Z") # Attempt local conversion
        except Exception: return dt_obj.strftime("%b %d, %Y %I:%M %p UTC") # Fallback UTC
    if not date_str or not isinstance(date_str, str): return "Date unavailable"
    try:
        dt_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        try: return dt_obj.astimezone().strftime("%b %d, %Y %I:%M %p %Z")
        except Exception: return dt_obj.strftime("%b %d, %Y %I:%M %p UTC")
    except (ValueError, TypeError): print(f"Warning: Could not parse date '{date_str}'."); return str(date_str)

def highlight_text(text, search_terms):
    """Highlights search terms in text (case-insensitive)."""
    if not text or not search_terms: return text
    highlighted_text = str(text)
    valid_terms = [term for term in search_terms if term and term.strip()]
    if not valid_terms: return highlighted_text
    for term in valid_terms:
        try:
            pattern = re.compile(re.escape(term.strip()), re.IGNORECASE)
            highlighted_text = pattern.sub(lambda m: f'<mark style="background-color:#FFFF00;font-weight:bold;">{m.group(0)}</mark>', highlighted_text)
        except re.error as e: print(f"Regex error highlighting term '{term}': {e}")
        except Exception as e: print(f"Error during highlighting term '{term}': {e}")
    return highlighted_text

def get_article_text(article):
    """Combines title and description for TF-IDF, ensuring non-empty string."""
    if not article or not isinstance(article, dict): return ""
    title = article.get('title', '') or ''; description = article.get('description', '') or ''
    return f"{str(title)} {str(description)}".strip()

def _hash_url(url):
    """Creates a SHA-1 hash of a URL for use as a Firestore document ID."""
    if not url or not isinstance(url, str): print("Warning: Attempted to hash invalid URL."); return None
    try: return hashlib.sha1(url.encode('utf-8')).hexdigest()
    except Exception as e: print(f"Error hashing URL '{url}': {e}"); return None

# --- API Interaction ---
@st.cache_data(ttl=600)
def fetch_news(query, page_size=DEFAULT_PAGE_SIZE, from_date=None, to_date=None, sort_by="relevancy", page=1, timeout=15):
    """Fetches news from NewsAPI /everything endpoint with specified timeout."""
    if not query or not query.strip(): print("fetch_news: empty query."); return []
    params = {'q': query.strip()[:500], 'apiKey': API_KEY, 'pageSize': min(page_size, 100),
              'sortBy': sort_by, "language": "en", "page": page}
    if from_date: params['from'] = from_date;
    if to_date: params['to'] = to_date
    print(f"Fetching NewsAPI: q='{params['q']}', sort={sort_by}, page={page}, from={from_date}, to={to_date}, timeout={timeout}s")
    try:
        response = requests.get(BASE_URL, params=params, timeout=timeout); response.raise_for_status()
        data = response.json()
        if data.get('status') == 'error':
            api_code=data.get('code','?'); api_msg=data.get('message','Error'); st.error(f"NewsAPI Error ({api_code}): {api_msg}"); print(f"NewsAPI Error: {api_code} - {api_msg} for q={params['q']}"); return []
        articles = [a for a in data.get('articles', []) if (a.get('title') and a.get('title') != '[Removed]' and a.get('url') and get_article_text(a))]
        print(f"NewsAPI OK: Found {len(articles)} articles."); return articles
    except requests.exceptions.Timeout: st.error(f"NewsAPI Timeout ({timeout}s). Try reducing filters."); print(f"NewsAPI Timeout: q='{params['q']}', t={timeout}s"); return None
    except requests.exceptions.HTTPError as e: st.error(f"HTTP Error fetching news: {e.response.status_code}"); print(f"NewsAPI HTTP Error: {e.response.status_code}"); return []
    except requests.exceptions.RequestException as e: st.error(f"Network error: {e}"); print(f"Network error: {e}"); return []
    except Exception as e: st.error(f"Unexpected news fetch error: {e}"); traceback.print_exc(); return []

@st.cache_data(ttl=120)
def fetch_top_headlines(category=None, country="us", page_size=1):
    """Fetches top headlines from NewsAPI /top-headlines endpoint."""
    params = {'apiKey': API_KEY, 'pageSize': min(page_size, 100), 'country': country}
    if category and category.lower() in NEWSAPI_TOP_HEADLINE_CATEGORIES: params['category'] = category.lower()
    elif category: print(f"Warning: Invalid category '{category}' for headlines.")
    try:
        response = requests.get(HEADLINES_URL, params=params, timeout=10); response.raise_for_status()
        data = response.json()
        if data.get('status') == 'error': print(f"NewsAPI Headlines Error: {data.get('message')}"); return None
        valid = [a for a in data.get('articles', []) if a.get('title') != '[Removed]' and a.get('url')]
        return valid[0] if page_size == 1 and valid else valid
    except Exception as e: print(f"Headline fetch error: {e}"); return None

# --- Firebase Interaction ---
def track_user_activity(user_id, article, action_type="view"):
    """Tracks user activity, storing likes/saves fully and logging others. Returns True on success, False on failure."""
    if not all([user_id, article, isinstance(article, dict), article.get('url')]): print(f"Warn: Invalid track_user_activity data ({action_type})."); return False
    article_url = article.get('url'); hashed_url_id = _hash_url(article_url)
    if not hashed_url_id: print(f"Warn: Could not hash URL for tracking: {article.get('title')}"); return False
    valid_actions = ["view", "save", "like", "dislike", "unsave", "unlike"]
    if action_type not in valid_actions: print(f"Warn: Invalid action type '{action_type}' for tracking."); return False

    try:
        user_ref = db.collection("users").document(user_id)
        if action_type == "save":
            article_data = article.copy(); article_data['saved_at'] = firestore.SERVER_TIMESTAMP
            if isinstance(article_data.get('source'), dict): article_data['source'] = {'name': article_data['source'].get('name')}
            article_data.pop('content', None)
            user_ref.collection("saved_articles").document(hashed_url_id).set(article_data)
            print(f"Saved '{hashed_url_id}' for {user_id}")
        elif action_type == "like":
            article_data = article.copy(); article_data['liked_at'] = firestore.SERVER_TIMESTAMP
            if isinstance(article_data.get('source'), dict): article_data['source'] = {'name': article_data['source'].get('name')}
            article_data.pop('content', None)
            user_ref.collection("liked_articles").document(hashed_url_id).set(article_data)
            print(f"Liked '{hashed_url_id}' for {user_id}")
            _log_generic_activity(user_ref, article, "like") # Log for recommendations
        elif action_type == "unsave":
            user_ref.collection("saved_articles").document(hashed_url_id).delete()
            print(f"Unsaved '{hashed_url_id}' for {user_id}")
        elif action_type == "unlike":
            user_ref.collection("liked_articles").document(hashed_url_id).delete()
            print(f"Unliked '{hashed_url_id}' for {user_id}")
            _log_generic_activity(user_ref, article, "dislike") # Log as dislike for recommendations
        elif action_type in ["view", "dislike"]:
             _log_generic_activity(user_ref, article, action_type) # Log views/dislikes for recommendations
        return True # Indicate success
    except Exception as e:
        st.error(f"Error processing '{action_type}' action: {e}") # Show error in UI
        print(f"Firestore error during '{action_type}' for user {user_id}, article {hashed_url_id}:")
        traceback.print_exc()
        return False # Indicate failure

def _log_generic_activity(user_ref, article, action_type):
    """Helper to log minimal activity info to the 'activity' subcollection."""
    try:
        activity_data = { "article_title": article.get('title'), "article_description": article.get('description'),
            "article_url": article.get('url'), "action_type": action_type, "timestamp": firestore.SERVER_TIMESTAMP,
            "article_source": article.get('source', {}).get('name'), "published_at": article.get('publishedAt'),
            "category_context": st.session_state.get('current_category'), "subcategory_context": st.session_state.get('current_subcategory'), }
        user_ref.collection("activity").add(activity_data)
    except Exception as e: print(f"Error logging generic activity '{action_type}' for {article.get('url')}: {e}"); traceback.print_exc()

def get_user_preferences(user_id):
    """Gets user preferences from Firestore."""
    if not user_id: return {}
    try:
        doc = db.collection("users").document(user_id).get()
        if doc.exists:
            prefs = doc.to_dict().get('preferences', {})
            prefs.setdefault('categories', [k for k in CATEGORIES if not k.startswith('_')])
            prefs.setdefault('notifications', False); return prefs
        else: print(f"Warn: User doc {user_id} not found."); return {'categories': [k for k in CATEGORIES if not k.startswith('_')], 'notifications': False}
    except Exception as e: st.error(f"Error get prefs: {e}"); traceback.print_exc(); return {}

def update_user_preferences(user_id, preferences):
    """Updates user preferences in Firestore."""
    if not user_id or not isinstance(preferences, dict): print("Warn: Invalid update prefs data."); return
    try: db.collection("users").document(user_id).set({'preferences': preferences, 'last_updated': firestore.SERVER_TIMESTAMP}, merge=True)
    except Exception as e: st.error(f"Error update prefs: {e}"); traceback.print_exc()

def get_user_activity(user_id, limit=50, action_types=None):
    """Retrieves generic activity log entries for recommendations."""
    if not user_id: return []
    try:
        q = (db.collection("users").document(user_id).collection("activity").order_by("timestamp", direction=firestore.Query.DESCENDING))
        # Requires composite index: activity(action_type ASC, timestamp DESC) if filtering
        if action_types and isinstance(action_types, list): q = q.where(filter=firestore.FieldFilter("action_type", "in", action_types))
        docs = q.limit(limit).stream(); activity_list = [doc.to_dict() for doc in docs]
        print(f"Retrieved {len(activity_list)} activity entries for {user_id} (filter: {action_types}).")
        return activity_list
    except exceptions.FailedPrecondition as e:
        st.error(f"Firestore Index Error: Check logs/console. Error: {e}")
        print(f"Firestore index missing for activity query ({action_types}) for {user_id}: {e}")
        match = re.search(r"(https://console\.firebase\.google\.com/.*)\)?$", str(e));
        if match: st.error(f"[Create Index]({match.group(1)})") # Use markdown link
        return []
    except Exception as e: st.error(f"Error getting activity: {e}"); traceback.print_exc(); return []

# --- Personalization / Recommendation ---
def get_personalized_recommendations(user_id, candidate_articles, activity_limit=100, min_recommendations=5, positive_interactions_threshold=3):
    """Recommends articles based on TF-IDF similarity to user's interaction history."""
    if not user_id or not candidate_articles:
        if candidate_articles: candidate_articles.sort(key=lambda x: x.get('publishedAt', ''), reverse=True)
        return (candidate_articles or [])[:DEFAULT_PAGE_SIZE]
    print(f"Starting personalized recommendations for {user_id} with {len(candidate_articles)} candidates.")
    interaction_history = get_user_activity(user_id, limit=activity_limit, action_types=['like', 'view', 'dislike'])
    if not interaction_history: print(f"{user_id}: No interaction history. Falling back."); candidate_articles.sort(key=lambda x: x.get('publishedAt',''), reverse=True); return candidate_articles[:DEFAULT_PAGE_SIZE]
    profile_actions={}; like_count=0
    for act in interaction_history:
        url=act.get('article_url'); text=get_article_text(act)
        if not url or not text: continue
        action=act.get('action_type'); weight=0.0
        if action=='like': weight=1.0; like_count+=1
        elif action=='view': weight=0.2 # Lower weight for views
        elif action=='dislike': weight=-1.0
        current_weight=profile_actions.get(url,{}).get('weight',0.0)
        if abs(weight)>=abs(current_weight): profile_actions[url]={'text':text,'weight':weight} # Keep strongest signal
    print(f"{user_id}: Processed {len(profile_actions)} unique articles from activity ({like_count} likes).")
    if like_count < positive_interactions_threshold: print(f"{user_id}: < {positive_interactions_threshold} likes. Falling back."); candidate_articles.sort(key=lambda x: x.get('publishedAt',''), reverse=True); return candidate_articles[:DEFAULT_PAGE_SIZE]
    profile_texts=[a['text'] for a in profile_actions.values()]; candidate_texts=[get_article_text(a) for a in candidate_articles]
    valid_profile_indices=[i for i,t in enumerate(profile_texts) if t]; valid_candidate_indices=[i for i,t in enumerate(candidate_texts) if t]
    if not valid_profile_indices or not valid_candidate_indices: print("Warn: No valid texts for TF-IDF."); candidate_articles.sort(key=lambda x: x.get('publishedAt',''), reverse=True); return candidate_articles[:DEFAULT_PAGE_SIZE]
    profile_texts_valid=[profile_texts[i] for i in valid_profile_indices]; profile_actions_valid=[list(profile_actions.values())[i] for i in valid_profile_indices]
    candidate_texts_valid=[candidate_texts[i] for i in valid_candidate_indices]; candidate_articles_valid=[candidate_articles[i] for i in valid_candidate_indices]
    all_texts=profile_texts_valid+candidate_texts_valid
    print(f"Running TF-IDF on {len(profile_texts_valid)} profile / {len(candidate_texts_valid)} candidate texts.")
    try:
        tfidf_matrix=vectorizer.fit_transform(all_texts); num_profile=len(profile_texts_valid); profile_vectors=tfidf_matrix[:num_profile]; candidate_vectors=tfidf_matrix[num_profile:]
        if profile_vectors.shape[0]<=0: raise ValueError("No profile vectors.")
        weighted_profile_sum=None
        for i,action in enumerate(profile_actions_valid): vector=profile_vectors[i]; weight=action['weight']; term=vector*weight; weighted_profile_sum=(term if weighted_profile_sum is None else weighted_profile_sum+term)
        user_profile_vector=(weighted_profile_sum if weighted_profile_sum is not None else profile_vectors.mean(axis=0)) # Fallback to mean
        user_profile_vector=np.asarray(user_profile_vector); user_profile_vector=user_profile_vector.reshape(1,-1) # Ensure correct shape
        if candidate_vectors.shape[0]>0: similarities=cosine_similarity(user_profile_vector,candidate_vectors).flatten(); print(f"Calculated {len(similarities)} similarity scores.")
        else: similarities=np.array([]); print("No candidates for similarity.")
        scored_candidates=[]; now_utc=datetime.now(timezone.utc)
        for i,article in enumerate(candidate_articles_valid):
            score=similarities[i] if i<len(similarities) else 0.0
            try: # Add recency boost
                pub_dt_str=article.get('publishedAt');
                if pub_dt_str: pub_dt=datetime.fromisoformat(pub_dt_str.replace('Z','+00:00')); days=max(0,(now_utc-pub_dt).days); boost=max(0,1.0-(days/14.0))*0.1; score+=boost
            except: pass # Ignore boost errors
            scored_candidates.append({'score':score,'article':article})
        scored_candidates.sort(reverse=True,key=lambda x:x['score']); recommended_articles=[item['article'] for item in scored_candidates]; print(f"Ranked {len(recommended_articles)} articles.")
        if len(recommended_articles)<min_recommendations: # Ensure minimum recommendations
            needed=min_recommendations-len(recommended_articles); rec_urls={a['url'] for a in recommended_articles};
            remaining=[a for a in candidate_articles_valid if a['url'] not in rec_urls]; remaining.sort(key=lambda x:x.get('publishedAt',''),reverse=True);
            recommended_articles.extend(remaining[:needed]); print(f"Added {needed} fallback articles.")
        return recommended_articles[:DEFAULT_PAGE_SIZE]
    except Exception as e: st.error(f"Error gen recommendations: {e}"); traceback.print_exc(); candidate_articles.sort(key=lambda x:x.get('publishedAt',''),reverse=True); return candidate_articles[:DEFAULT_PAGE_SIZE]

# --- UI Components ---

def display_article(article, user_id=None, search_terms=None, context="feed"):
    """Displays a single article card with context-aware interaction buttons."""
    if not article or not isinstance(article, dict) or not article.get('url'):
         print("Warning: display_article received invalid article data.")
         return

    url = article.get('url')
    hashed_url = _hash_url(url)
    unique_key_base = f"{context}_{hashed_url}" if hashed_url else f"{context}_{url[:50]}" # Use hashed URL for key

    with st.container():
        col1, col2 = st.columns([1, 3], gap="medium") # Image | Text layout

        with col1: # Image Column
            image_url = article.get('urlToImage')
            st.image(image_url or PLACEHOLDER_IMAGE, use_container_width=True,
                     caption=f"Source: {article.get('source', {}).get('name', 'Unknown')}")

        with col2: # Text & Interaction Column
            title = article.get('title', 'No Title Provided')
            description = article.get('description') or 'No description available.'
            title_html = highlight_text(title, search_terms) if search_terms else title
            desc_raw = description # Keep original description for length check

            # Display Title
            if search_terms: st.markdown(f"<h4>{title_html}</h4>", unsafe_allow_html=True)
            else: st.subheader(title)

            # Display Metadata (Author, Date - context aware)
            metadata = []; author = article.get('author')
            if author and isinstance(author, str): metadata.append(f"By {author[:50]}{'...' if len(author)>50 else ''}")
            date_fmt, date_lbl = None, None
            if context == 'save' and 'saved_at' in article: date_fmt, date_lbl = article['saved_at'], "Saved:"
            elif context == 'like' and 'liked_at' in article: date_fmt, date_lbl = article['liked_at'], "Liked:"
            else: date_fmt, date_lbl = article.get('publishedAt'), "Published:"
            fmt_date = format_date(date_fmt); metadata.append(f"{date_lbl} {fmt_date}" if date_lbl else fmt_date)
            st.caption(" • ".join(filter(None, metadata)))

            # Display Description (Truncated & Highlighted if needed)
            max_len = 250;
            if len(desc_raw) > max_len: desc_display = desc_raw[:max_len] + "..."
            else: desc_display = desc_raw
            st.markdown(f"<p>{highlight_text(desc_display, search_terms) if search_terms else desc_display}</p>", unsafe_allow_html=True)

            # Link to Full Article
            st.markdown(f"[Read full article]({url})", unsafe_allow_html=True)

            # --- Interaction Buttons (Context Aware Like/Unlike, Save/Unsave) ---
            if user_id:
                btn_col1, btn_col2, btn_col3, _ = st.columns([1, 1, 1, 3]) # Button layout

                # Like/Unlike Button
                with btn_col1:
                    is_liked = (context == 'like') # Determine based on context
                    like_btn_text = "💔 Unlike" if is_liked else "👍 Like"
                    like_btn_action = "unlike" if is_liked else "like"
                    like_btn_help = "Remove from liked" if is_liked else "Like this article"
                    if st.button(like_btn_text, key=f"like_{unique_key_base}", help=like_btn_help):
                        success = track_user_activity(user_id, article, like_btn_action)
                        if success:
                            st.toast("Unliked!" if is_liked else "Liked!", icon="💔" if is_liked else "👍")
                            if is_liked: time.sleep(0.5); st.rerun() # Rerun to refresh Liked list
                        else: st.error(f"Failed to {like_btn_action}.")

                # Dislike Button (Always shown)
                with btn_col2:
                    if st.button("👎 Dislike", key=f"dislike_{unique_key_base}", help="Dislike article"):
                         if track_user_activity(user_id, article, "dislike"): st.toast("Noted!", icon="👎")
                         else: st.error("Failed to record dislike.")

                # Save/Unsave Button
                with btn_col3:
                    is_saved = (context == 'save') # Determine based on context
                    save_btn_text = "❌ Unsave" if is_saved else "💾 Save"
                    save_btn_action = "unsave" if is_saved else "save"
                    save_btn_help = "Remove from saved" if is_saved else "Save for later"
                    if st.button(save_btn_text, key=f"save_{unique_key_base}", help=save_btn_help):
                        success = track_user_activity(user_id, article, save_btn_action)
                        if success:
                            st.toast("Unsaved!" if is_saved else "Saved!", icon="❌" if is_saved else "💾")
                            if is_saved: time.sleep(0.5); st.rerun() # Rerun to refresh Saved list
                        else: st.error(f"Failed to {save_btn_action}.")
            else: st.caption("Login/Sign Up to interact.")

        # Implicit View Tracking (only for feed/search contexts)
        if user_id and context in ["feed", "advanced_search"]:
             if 'view_tracked' not in st.session_state: st.session_state.view_tracked = set()
             if url not in st.session_state.view_tracked:
                 track_user_activity(user_id, article, "view"); st.session_state.view_tracked.add(url)

        st.markdown("---") # Visual separator

def auth_component():
    """Handles user authentication in the sidebar."""
    if 'user' not in st.session_state: st.session_state.user = None
    with st.sidebar:
        st.markdown(f"""
        <h1 style="
            font-size: 2.8em; /* Make it significantly larger */
            font-weight: 700; /* Bolder */
            color: #007bff; /* A distinct blue color (adjust as desired) */
            text-align: center; /* Center the text */
            margin-bottom: 0.5em; /* Add some space below */
            padding-top: 0px; /* Adjust top padding if needed */
            font-family: 'Arial Black', Gadget, sans-serif; /* Example bold font */
        ">
            NEWSIFY
        </h1>
        """, unsafe_allow_html=True)

        st.title("Account")
        if st.session_state.user:
            st.write(f"Welcome, **{st.session_state.user['email']}**")
            if st.button("Sign Out", key="signout_btn_key"):
                email = st.session_state.user['email']; st.session_state.user = None
                keys = list(st.session_state.keys());
                for key in keys:
                    if key not in ['firebase_initialized']: st.session_state.pop(key, None) # Clear state
                st.toast(f"Signed out {email}"); st.rerun()
        else:
            choice = st.selectbox("Login / Sign Up", ["Login", "Sign Up"], label_visibility="collapsed", key="auth_choice_key")
            form_key = f"auth_form_{choice.lower()}_key"
            with st.form(form_key, clear_on_submit=False):
                email = st.text_input("Email", key=f"{form_key}_email_wid")
                pwd = st.text_input("Password", type="password", key=f"{form_key}_pwd_wid")
                submitted = st.form_submit_button(choice)
                if submitted:
                    if not email or not pwd: st.error("Email/Pwd required."); return
                    if not re.match(r"[^@]+@[^@]+\.[^@]+", email): st.error("Valid email."); return
                    if choice == "Login":
                        try:
                            user = auth.get_user_by_email(email); st.warning("DEMO LOGIN: Pwd NOT verified.")
                            st.session_state.user = {'email': user.email, 'uid': user.uid}
                            if EMAIL_CONFIGURED: send_login_notification(user.email)
                            st.toast("Login OK!"); st.rerun()
                        except exceptions.NotFoundError: st.error("Login Failed: User not found.")
                        except Exception as e: st.error(f"Login Failed: {e}"); traceback.print_exc()
                    elif choice == "Sign Up":
                        if len(pwd) < 6: st.error("Pwd >= 6 chars."); return
                        try:
                            user = auth.create_user(email=email, password=pwd, email_verified=False); print(f"User created: {user.uid}")
                            st.success(f"Account created! Please log in.")
                            try: default_prefs = {'categories': [k for k in CATEGORIES if not k.startswith('_')], 'notifications': False}; db.collection("users").document(user.uid).set({'email': email, 'preferences': default_prefs, 'created_at': firestore.SERVER_TIMESTAMP})
                            except Exception as db_e: print(f"Error creating profile: {db_e}"); st.warning("Account OK, profile init failed.")
                            if EMAIL_CONFIGURED: send_signup_notification(email)
                        except exceptions.EmailAlreadyExistsError: st.error("Email already registered.")
                        except Exception as e: st.error(f"Sign Up Failed: {e}"); traceback.print_exc()

def preferences_component(user_id):
    """Component for managing user preferences in the sidebar."""
    if not user_id: return
    with st.sidebar, st.expander("Manage Preferences", expanded=False):
        st.markdown("---"); st.subheader("Your Preferences")
        prefs = get_user_preferences(user_id)
        all_keys = [k for k in CATEGORIES if not k.startswith('_')]; map_d = {k: CATEGORIES[k]['display_name'] for k in all_keys}
        curr_keys = prefs.get('categories', all_keys); curr_d = [map_d[k] for k in curr_keys if k in map_d]
        pref_d = st.multiselect("Categories", list(map_d.values()), default=curr_d, key="pref_cats_wid_key")
        map_k = {v: k for k, v in map_d.items()}; pref_k = [map_k[d] for d in pref_d if d in map_k]
        notif_on = st.checkbox("Show breaking news", value=prefs.get('notifications', False), key="pref_notif_wid_key")
        if st.button("Save Preferences", key="save_prefs_wid_btn_key"):
            update_user_preferences(user_id, {'categories': pref_k, 'notifications': notif_on})
            st.success("Saved!"); st.cache_data.clear(); time.sleep(1); st.rerun()

def notification_panel_component(user_id):
    """Displays a periodic breaking news notification panel if enabled."""
    if not user_id: return
    prefs = get_user_preferences(user_id)
    if not prefs.get('notifications', False): return
    CHECK_INT = 180
    if 'last_notif_check' not in st.session_state: st.session_state.last_notif_check = 0
    if 'curr_notif_article' not in st.session_state: st.session_state.curr_notif_article = None
    now = time.time()
    if now - st.session_state.last_notif_check > CHECK_INT:
        print("Checking notifications...")
        st.session_state.last_notif_check = now; latest = get_latest_breaking_news_for_user(user_id)
        curr_url = st.session_state.curr_notif_article.get('url') if st.session_state.curr_notif_article else None
        new_url = latest.get('url') if latest else None
        if new_url and new_url != curr_url: st.session_state.curr_notif_article = latest; print(f"New notif: {new_url}"); st.rerun()
        elif not new_url: print("No new breaking news.")
    notif = st.session_state.curr_notif_article
    if notif:
        with st.container(): # Apply CSS locally
             st.markdown("""<style>.notification-panel{border:1px solid #dee2e6;border-left:5px solid #17a2b8;border-radius:.375rem;padding:.8rem 1rem;background-color:#e6f7ff;margin-bottom:1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.1);font-size:.9em}.notification-panel p{margin-bottom:.25rem}.notification-title a{color:#0056b3!important;font-weight:600;text-decoration:none}.notification-title a:hover{text-decoration:underline;color:#003d80!important}.notification-caption{font-size:.9em;color:#6c757d}</style>""", unsafe_allow_html=True)
             st.markdown('<div class="notification-panel">', unsafe_allow_html=True)
             title=notif.get('title','No Title'); url=notif.get('url','#'); src=notif.get('source',{}).get('name','Unknown'); pub=format_date(notif.get('publishedAt'))
             st.markdown(f"<p class='notification-title'>🔔  <b>Breaking:</b> <a href='{url}' target='_blank' title='{title}'>{title}</a></p>", unsafe_allow_html=True)
             st.markdown(f"<p class='notification-caption'>Source: {src} | Published: {pub}</p>", unsafe_allow_html=True)
             st.markdown('</div>', unsafe_allow_html=True)

def get_latest_breaking_news_for_user(user_id):
    """Helper to get the single latest breaking news article across user's preferred categories."""
    if not user_id: return None
    prefs = get_user_preferences(user_id); pref_keys = prefs.get('categories', [])
    if not pref_keys: return None
    valid_cats = [k for k in pref_keys if k in NEWSAPI_TOP_HEADLINE_CATEGORIES]
    if not valid_cats: return None
    latest_a = None; latest_ts = datetime.min.replace(tzinfo=timezone.utc)
    for cat in valid_cats:
        article = fetch_top_headlines(category=cat, page_size=1, country="us")
        if article and article.get('publishedAt'):
            try: pub_ts = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
            except: continue
            if pub_ts > latest_ts: latest_ts = pub_ts; latest_a = article
    return latest_a

# --- Main Application Structure ---

def display_regular_news_feed(user_id):
    """Displays the main news feed tab, handling different feed types."""
    # --- Sidebar Setup ---
    # (Handles selecting category, subcategory, date, search, sets flags/variables)
    sidebar_from_date, sidebar_to_date = None, None; selected_category_key = None; selected_subcategory_name = None
    search_term_from_sidebar = None; category_display_name = "News"; is_personalized_feed = False; is_saved_liked_feed = False
    action_type_to_fetch = None; feed_context = "feed"

    with st.sidebar:
        st.subheader("Filter by Date"); date_opts = ["24h", "3 Days", "Week", "Month", "All Time"]; date_map = {"24h": 1, "3 Days": 3, "Week": 7, "Month": 30}
        date_choice = st.radio("Time", date_opts, index=3, key="feed_date_wid_key", label_visibility="collapsed", horizontal=True)
        now = datetime.now(timezone.utc);
        if date_choice != "All Time": days = date_map[date_choice]; sidebar_from_date = (now - timedelta(days=days)).strftime('%Y-%m-%d'); sidebar_to_date = now.strftime('%Y-%m-%d')

        st.markdown("---"); st.subheader("Categories")
        all_keys = list(CATEGORIES.keys());
        if not user_id: all_keys = [k for k in all_keys if not k.startswith('_')]
        prefs = get_user_preferences(user_id); pref_keys = prefs.get('categories', [])
        avail_keys = [];
        if user_id: avail_keys.append('_personalized_'); avail_keys.extend(sorted([k for k in pref_keys if not k.startswith('_') and k in CATEGORIES]))
        else: avail_keys.extend(sorted([k for k in all_keys if not k.startswith('_') and k in CATEGORIES]))
        avail_keys = sorted(list(set(avail_keys)), key=lambda k: (k != '_personalized_', k)) # Keep personalized first
        cat_map = {k: f"{CATEGORIES[k]['emoji']} {CATEGORIES[k]['display_name']}" for k in avail_keys if k in CATEGORIES}
        if 'feed_cat_wid_key' not in st.session_state or st.session_state.feed_cat_wid_key not in avail_keys: st.session_state.feed_cat_wid_key = avail_keys[0] if avail_keys else None
        selected_category_key = st.radio("Category:", avail_keys, format_func=lambda k: cat_map.get(k, k.replace('_',' ').capitalize()), key="feed_cat_wid_key", label_visibility="collapsed")

        if not selected_category_key: st.warning("No category selected."); return
        category_display_name = CATEGORIES[selected_category_key]['display_name']; st.session_state.current_category = selected_category_key

        if selected_category_key in CATEGORIES and 'subcategories' in CATEGORIES[selected_category_key]:
            st.subheader(f"{category_display_name} Topics"); subcat_map = CATEGORIES[selected_category_key]["subcategories"]; subcat_names = list(subcat_map.keys())
            subcat_key = f"feed_subcat_{selected_category_key}_wid_key" # Unique key
            if subcat_key not in st.session_state: st.session_state[subcat_key] = subcat_names[0] if subcat_names else None
            selected_subcategory_name = st.radio("Topic:", subcat_names, key=subcat_key, label_visibility="collapsed")
            st.session_state.current_subcategory = selected_subcategory_name
            search_term_from_sidebar = subcat_map.get(selected_subcategory_name)
            if selected_category_key == '_personalized_':
                 is_personalized_feed = True; action_map = {'Saved Articles': 'save', 'Liked Articles': 'like'}
                 if selected_subcategory_name in action_map: is_saved_liked_feed = True; action_type_to_fetch = action_map[selected_subcategory_name]; category_display_name = f"My {selected_subcategory_name}"; feed_context = action_type_to_fetch
                 else: category_display_name = "News For You"
                 search_term_from_sidebar = None
            else: category_display_name = f"{category_display_name}: {selected_subcategory_name}"
        else: search_term_from_sidebar = selected_category_key.lower(); st.session_state.current_subcategory = None

        st.markdown("---"); st.subheader("Quick Search")
        custom_search = st.text_input("🔍 Search:", key="q_search_wid_key", placeholder="Keywords...", label_visibility="collapsed", help="Overrides category").strip()
        preferences_component(user_id) # Show preferences management

    # --- Determine Final Search & Display Settings ---
    final_search_term = None; articles = []; fetch_error = False
    is_custom_search = bool(custom_search); final_from = sidebar_from_date; final_to = sidebar_to_date
    highlight_terms = []; color = CATEGORIES.get(selected_category_key, {}).get('color', "#6c757d"); emoji = CATEGORIES.get(selected_category_key, {}).get('emoji', "📰")

    if is_custom_search: final_search_term = custom_search; category_display_name = f"Search: '{custom_search}'"; color, emoji = "#6f42c1", "🔍"; highlight_terms = custom_search.split(); feed_context = "advanced_search"
    elif is_saved_liked_feed: color = CATEGORIES['_personalized_']["color"]; emoji = "💾" if action_type_to_fetch == 'save' else "👍"
    elif is_personalized_feed: color = CATEGORIES['_personalized_']["color"]; emoji = CATEGORIES['_personalized_']["emoji"]
    else: final_search_term = search_term_from_sidebar

    # --- Display Banner & Refresh Button ---
    st.markdown(f"""<div style="background:linear-gradient(to right,{color},{color}dd);color:#fff;padding:.8rem 1.5rem;border-radius:.5rem;margin-bottom:1.5rem;text-align:center;box-shadow:0 2px 4px rgba(0,0,0,.1)"><h2 style="color:#fff;margin:0;font-weight:700;font-size:1.5em">{emoji} {category_display_name}</h2></div>""", unsafe_allow_html=True)
    if st.button("🔄 Refresh Feed", key="refresh_top_wid_btn", help="Reload articles"): st.cache_data.clear(); st.rerun()

    # --- Fetch/Retrieve Articles (Main Logic) ---
    with st.spinner(f"Loading {category_display_name}..."):
        try:
            if is_saved_liked_feed: # Retrieve from Firestore
                collection = f"{action_type_to_fetch}d_articles"; ts_field = f"{action_type_to_fetch}d_at"
                docs_q = (db.collection("users").document(user_id).collection(collection).order_by(ts_field, direction=firestore.Query.DESCENDING).limit(100))
                articles = [doc.to_dict() for doc in docs_q.stream()]
                if not articles: st.info(f"You haven't {action_type_to_fetch}d any articles yet.") # Show message here
                else: print(f"Retrieved {len(articles)} {action_type_to_fetch}d articles from Firestore.")

            elif is_personalized_feed: # Fetch candidates -> Recommend
                print("Fetching candidates for 'For You'..."); prefs = get_user_preferences(user_id)
                pref_keys_q = prefs.get('categories', ['general', 'technology', 'business'])
                cand_q = " OR ".join([f'"{k}"' for k in pref_keys_q if k]) if pref_keys_q else "news"
                cand_articles = fetch_news(query=cand_q, page_size=50, from_date=final_from, to_date=final_to, sort_by="popularity", timeout=25)
                if cand_articles is None: fetch_error = True; articles = [] # Handle timeout
                elif not cand_articles: print("No candidates found for 'For You'."); articles = []
                else: print(f"Generating recommendations from {len(cand_articles)} candidates..."); articles = get_personalized_recommendations(user_id, cand_articles)

            elif final_search_term: # Fetch from NewsAPI
                print(f"Fetching news for: {final_search_term}");
                articles = fetch_news(query=final_search_term, page_size=DEFAULT_PAGE_SIZE, from_date=final_from, to_date=final_to, sort_by="relevancy" if is_custom_search else "publishedAt")
                if articles is None: fetch_error = True; articles = [] # Handle timeout

            else: st.warning("Please select a category or enter a search term."); articles = []

        except Exception as e: st.error(f"Error processing feed: {e}"); traceback.print_exc(); articles = []; fetch_error = True

    # --- Display Articles or Fallback Messages ---
    if fetch_error: st.error(f"Could not load articles for '{category_display_name}'. News source might be unavailable. Please try again later.")
    elif not articles:
        if not is_saved_liked_feed: # Avoid duplicate message
             st.warning("No articles found matching your criteria.")
             if is_personalized_feed: st.info("Interact with articles (like, save, view) to improve 'For You' recommendations.")
    else: # Display articles
        st.success(f"Displaying {len(articles)} articles.")
        if feed_context in ["feed", "advanced_search"]: st.session_state.view_tracked = set() # Reset view tracking
        for article in articles: display_article(article, user_id, search_terms=highlight_terms if is_custom_search else None, context=feed_context)
        if st.button("🔄 Refresh Feed", key="refresh_bottom_wid_btn", help="Reload articles"): st.cache_data.clear(); st.rerun()

def display_advanced_search_interface(user_id):
    """Displays the advanced search tab content."""
    st.subheader("🔍 Advanced Article Search")
    st.write("Refine search across news (last ~30 days). Use NewsAPI [syntax](https://newsapi.org/docs/endpoints/everything#searchIn) (e.g., `+keyword`, `\"phrase\"`).")
    with st.form(key="adv_s_form_key"):
        search_q = st.text_input("**Keywords/Query**", placeholder="(apple OR google) AND innovation NOT lawsuit", key="adv_q_wid_key")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Categories (Opt.)**"); all_d= [CATEGORIES[k]['display_name'] for k in CATEGORIES if not k.startswith('_')]
            sel_d = st.multiselect("Limit to:", all_d, key="adv_cats_wid_key", label_visibility="collapsed")
            map_k = {CATEGORIES[k]['display_name']: k for k in CATEGORIES if not k.startswith('_')}; sel_k = [map_k[n] for n in sel_d if n in map_k]
        with c2:
            st.markdown("**Time Period**"); d_opts = ["24h", "7d", "30d", "Custom"]; d_sel = st.selectbox("Select:", d_opts, index=2, key="adv_date_wid_key", label_visibility="collapsed")
            from_d, to_d = None, None; now = datetime.now(timezone.utc)
            if d_sel == "Custom": min_d=now-timedelta(days=30); f_in=st.date_input("From",key="adv_s_dt_key",value=min_d,min_value=min_d,max_value=now); t_in=st.date_input("To",key="adv_e_dt_key",value=now,min_value=f_in,max_value=now); from_d=f_in.strftime('%Y-%m-%d'); to_d=t_in.strftime('%Y-%m-%d')
            else: map_d={"24h":1,"7d":7,"30d":30}; from_d=(now-timedelta(days=map_d[d_sel])).strftime('%Y-%m-%d'); to_d=now.strftime('%Y-%m-%d')
        sort_by = st.selectbox("**Sort by**", ["relevancy", "popularity", "publishedAt"], index=0, key="adv_sort_wid_key")
        submitted = st.form_submit_button("Search Articles")
    if submitted:
        if not search_q or not search_q.strip(): st.warning("Enter keywords."); return
        with st.spinner(f"Searching..."):
            final_q = search_q.strip()
            if sel_k: terms=[]; [terms.extend([k]+[t for t in CATEGORIES.get(k,{}).get('subcategories',{}).values() if isinstance(t,str) and not t.startswith('_')]) for k in sel_k]; final_q=f"({final_q}) AND ({' OR '.join(list(set(terms)))})"
            print(f"Adv Search: Q='{final_q}', From={from_d}, To={to_d}, Sort={sort_by}")
            articles = fetch_news(query=final_q, page_size=50, from_date=from_d, to_date=to_d, sort_by=sort_by) # Default timeout
            if articles is None: st.error("Advanced search failed or timed out.")
            elif not articles: st.warning("No articles found matching criteria.")
            else:
                st.success(f"Found {len(articles)} articles."); highlights = search_q.split(); st.session_state.view_tracked = set()
                for article in articles: display_article(article, user_id, search_terms=highlights, context="advanced_search")

# --- Main App Container ---
def main_app(user_id):
    """Main application container for logged-in users."""
    st.title("📰 Personalized News Explorer")
    notification_panel_component(user_id) # Display breaking news if enabled
    tab1, tab2 = st.tabs(["📰 News Feed", "🔍 Advanced Search"])
    with tab1: display_regular_news_feed(user_id) # Handles all feed types
    with tab2: display_advanced_search_interface(user_id)

# --- Main Execution Logic ---
def main():
    """Sets up the app, handles initialization checks, auth, and content display."""
    print("\n--- Streamlit App Execution Start ---") # Log start
    # --- Initial Checks ---
    if not st.session_state.get("firebase_initialized", False): st.error("Fatal Error: Firebase not initialized."); st.stop()
    if not API_KEY: st.error("Fatal Error: NewsAPI Key missing."); st.stop()
    try: auth.list_users(max_results=1) # Quick check post-init
    except Exception as e: st.error(f"Fatal Error: Firebase connection issue post-init: {e}"); st.stop()

    # --- Authentication ---
    auth_component() # Sidebar component handles login/signup/logout

    # --- Conditional Content Display ---
    if st.session_state.get('user') and st.session_state.user.get('uid'):
        main_app(st.session_state.user['uid']) # User is logged in
    else: # User is not logged in
        st.info("👋 Welcome! Please **Login** or **Sign Up** (sidebar) for personalized features.")
        st.markdown("---"); st.subheader("🇺🇸 Top US Headlines")
        with st.spinner("Loading headlines..."): trending = fetch_top_headlines(country='us', page_size=5)
        if trending: # Display generic headlines
             for article in trending[:3]:
                 if article:
                     with st.container(): c1, c2 = st.columns([1, 4]);
                     with c1: img = article.get('urlToImage'); st.image(img or PLACEHOLDER_IMAGE, use_container_width=True)
                     with c2: st.markdown(f"**[{article.get('title', 'No Title')}]({article.get('url')})**"); st.caption(f"{article.get('source',{}).get('name','Unknown')} • {format_date(article.get('publishedAt'))}"); desc = article.get('description') or ''; desc = (desc[:150] + '...') if len(desc) > 150 else desc; st.write(desc)
                     st.markdown("---")
        else: st.warning("Could not load trending headlines.")

    # --- Footer ---
    st.markdown("---")
    st.caption("Powered by [NewsAPI.org](https://newsapi.org) | Built with [Streamlit](https://streamlit.io)")
    st.caption("Note: Demo Login does not verify password.")

# --- Script Execution Entry Point ---
if __name__ == "__main__":
    main()
# --- END OF FILE news_app.py ---