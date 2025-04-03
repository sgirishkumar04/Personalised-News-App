import streamlit as st
st.set_page_config(
    layout="wide",
    page_title="Personalized News Aggregator",
    page_icon="📰"
)

import requests
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, auth, firestore, exceptions
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import re
from streamlit.components.v1 import html
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl # Added for improved SMTP TLS

# Load environment variables
load_dotenv()

# Configuration
# Use st.secrets for deployment, fallback to os.getenv for local dev
API_KEY = os.getenv("NEWSAPI_KEY") or st.secrets.get("NEWSAPI_KEY")
FIREBASE_CONFIG = {
    "type": os.getenv("FIREBASE_TYPE") or st.secrets.get("FIREBASE_TYPE"),
    "project_id": os.getenv("FIREBASE_PROJECT_ID") or st.secrets.get("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID") or st.secrets.get("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": (os.getenv("FIREBASE_PRIVATE_KEY") or st.secrets.get("FIREBASE_PRIVATE_KEY","")).replace('\\n', '\n'),
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL") or st.secrets.get("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID") or st.secrets.get("FIREBASE_CLIENT_ID"),
    "auth_uri": os.getenv("FIREBASE_AUTH_URI") or st.secrets.get("FIREBASE_AUTH_URI"),
    "token_uri": os.getenv("FIREBASE_TOKEN_URI") or st.secrets.get("FIREBASE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_CERT_URL") or st.secrets.get("FIREBASE_AUTH_PROVIDER_CERT_URL"),
    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL") or st.secrets.get("FIREBASE_CLIENT_CERT_URL")
}

# Email configuration - using environment variables or Streamlit secrets
SMTP_SERVER = os.getenv("SMTP_SERVER") or st.secrets.get("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT") or st.secrets.get("SMTP_PORT", 587))
SMTP_USERNAME = os.getenv("SMTP_USERNAME") or st.secrets.get("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD") or st.secrets.get("SMTP_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM") or st.secrets.get("EMAIL_FROM")
APP_NAME = "Personalized News Aggregator"

# Check if essential Firebase config is present
if not all(FIREBASE_CONFIG.values()):
    st.error("Firebase configuration is missing. Please check your .env file or Streamlit secrets.")
    st.stop()

# Check if email configuration is complete
EMAIL_CONFIGURED = all([SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, EMAIL_FROM])
if not EMAIL_CONFIGURED and not os.getenv("IS_LOCAL_DEV"): # Only warn prominently if not explicitly local dev
     print("Warning: Email configuration is incomplete. Notifications will be disabled.")
     # Optionally show a less intrusive warning in the UI if needed later


BASE_URL = "https://newsapi.org/v2/everything"
HEADLINES_URL = "https://newsapi.org/v2/top-headlines"
DEFAULT_PAGE_SIZE = 15
PLACEHOLDER_IMAGE = "https://via.placeholder.com/400x200?text=No+Image+Available"

# Initialize Firebase
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CONFIG)
        firebase_admin.initialize_app(cred)
        st.session_state.firebase_initialized = True
except ValueError as e:
     st.error(f"Failed to initialize Firebase: Invalid credentials format. Please check your FIREBASE_PRIVATE_KEY. Error: {e}")
     st.session_state.firebase_initialized = False
     st.stop()
except Exception as e:
    st.error(f"Failed to initialize Firebase: {str(e)}")
    st.session_state.firebase_initialized = False
    st.stop()

# Verify Firestore connection only if initialization seemed okay
if st.session_state.get("firebase_initialized", False):
    try:
        db = firestore.client()
        # Test connection by trying to access a non-existent doc
        db.collection('_test_connection').document('_test').get()
    except exceptions.PermissionDeniedError:
        st.error("Firestore permissions error. Please check your Firebase Security Rules.")
        st.stop()
    except exceptions.FailedPreconditionError as e:
        st.error(f"Firestore not enabled or missing index: {str(e)}")
        st.markdown(
            f"""
            Ensure Firestore is enabled in your Firebase project and Native mode is selected.
            <a href="https://console.firebase.google.com/project/{FIREBASE_CONFIG.get('project_id','_')}/firestore" target="_blank">
                Go to Firestore Console
            </a>
            """,
            unsafe_allow_html=True
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to connect to Firestore: {str(e)}")
        st.stop()
else:
     st.stop()

# Categories with subcategories
CATEGORIES = {
    '_personalized_': {
        'display_name': '✨ Personalized',
        'emoji': '✨',
        'color': '#6f42c1', # Bootstrap purple
        'subcategories': {
            'For You': '_for_you_',
            'Saved Articles': '_saved_',
            'Liked Articles': '_liked_',
        }
    },
    'technology': {
        'display_name': 'Technology',
        'emoji': '💻',
        'color': '#007bff', # Bootstrap primary blue
        'subcategories': {
            'General Tech': 'technology',
            'AI & Machine Learning': '"artificial intelligence" OR "machine learning"',
            'Gadgets': 'gadgets OR consumer electronics',
            'Software Development': '"software development" OR programming OR coding',
            'Cybersecurity': 'cybersecurity OR hacking OR "data breach"',
            'Startups & VC': 'startup OR venture capital OR funding',
        }
    },
    'business': {
        'display_name': 'Business',
        'emoji': '💼',
        'color': '#28a745', # Bootstrap success green
        'subcategories': {
            'General Business': 'business',
            'Markets': 'stock market OR finance OR investing',
            'Economy': 'economy OR inflation OR GDP',
            'Corporate News': 'corporate earnings OR mergers OR acquisitions',
            'Personal Finance': '"personal finance" OR budgeting OR saving',
        }
    },
    'science': {
        'display_name': 'Science',
        'emoji': '🔬',
        'color': '#17a2b8', # Bootstrap info cyan
        'subcategories': {
            'General Science': 'science',
            'Space & Astronomy': 'space OR astronomy OR NASA OR SpaceX',
            'Environment': 'environment OR climate change OR conservation',
            'Physics': 'physics OR quantum',
            'Biology': 'biology OR genetics OR evolution',
        }
    },
    'health': {
        'display_name': 'Health',
        'emoji': '❤️',
        'color': '#dc3545', # Bootstrap danger red
        'subcategories': {
            'General Health': 'health',
            'Medicine & Research': 'medical research OR clinical trial OR disease',
            'Wellness & Fitness': 'wellness OR fitness OR nutrition OR mental health',
            'Healthcare Policy': '"healthcare policy" OR insurance OR FDA',
        }
    },
    'sports': {
        'display_name': 'Sports',
        'emoji': '⚽',
        'color': '#ffc107', # Bootstrap warning yellow
        'subcategories': {
            'General Sports': 'sports',
            'Football (Soccer)': 'football OR soccer OR premier league OR champions league',
            'Basketball': 'basketball OR NBA',
            'American Football': '"american football" OR NFL',
            'Tennis': 'tennis OR ATP OR WTA',
            'Olympics': 'olympics',
        }
    },
    'entertainment': {
        'display_name': 'Entertainment',
        'emoji': '🎬',
        'color': '#fd7e14', # Bootstrap orange
        'subcategories': {
            'General Entertainment': 'entertainment',
            'Movies': 'movie OR film OR cinema',
            'Music': 'music OR artist OR concert',
            'Television': 'television OR TV show OR streaming',
            'Gaming': 'video game OR gaming OR esports',
            'Celebrity': 'celebrity OR hollywood',
        }
    },
    'general': {
        'display_name': 'General',
        'emoji': '📰',
        'color': '#6c757d', # Bootstrap secondary grey
        'subcategories': {
            'World News': 'world news',
            'US News': '"US news" OR "United States"',
            'Politics': 'politics',
            'Culture': 'culture OR arts',
        }
    }
}


# Valid categories for NewsAPI top-headlines endpoint
NEWSAPI_TOP_HEADLINE_CATEGORIES = {'business', 'entertainment', 'general', 'health', 'science', 'sports', 'technology'}

# Recommendation system setup (placeholder, could be enhanced)
vectorizer = TfidfVectorizer(stop_words='english')

# --- Email Notification Functions ---

def send_email(to_email, subject, body, is_html=False):
    """Send an email using SMTP with improved error handling and SSL context."""
    if not EMAIL_CONFIGURED:
        # This function should ideally not be called if not configured,
        # but this is a safety check.
        print("Error: send_email called but email is not configured.")
        return False

    try:
        # Create message container
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = to_email
        msg['Subject'] = subject

        # Attach the body
        if is_html:
            msg.attach(MIMEText(body, 'html'))
        else:
            msg.attach(MIMEText(body, 'plain'))

        # Create secure connection with server and send email
        context = ssl.create_default_context() # Use default SSL context

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()  # Optional, but good practice
            server.starttls(context=context) # Secure the connection
            server.ehlo()  # Optional, re-identify after starting TLS
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

        return True

    except smtplib.SMTPAuthenticationError:
        st.error("Email failed: Authentication error. Please check your email credentials.")
        print(f"SMTP Authentication Error for user {SMTP_USERNAME} on {SMTP_SERVER}:{SMTP_PORT}")
        return False
    except smtplib.SMTPServerDisconnected:
        st.error("Email failed: Server disconnected unexpectedly.")
        print(f"SMTP Server Disconnected: {SMTP_SERVER}:{SMTP_PORT}")
        return False
    except smtplib.SMTPException as e:
        st.error(f"Email failed: An SMTP error occurred: {str(e)}")
        print(f"SMTP Error: {e}")
        return False
    except TimeoutError:
         st.error("Email failed: Connection to the email server timed out.")
         print(f"SMTP Timeout Error: {SMTP_SERVER}:{SMTP_PORT}")
         return False
    except Exception as e:
        st.error(f"Unexpected error sending email: {str(e)}")
        print(f"Unexpected Email Error: {e}")
        return False

def send_login_notification(email):
    """Send notification email after successful login."""
    # Check configuration *before* proceeding
    if not EMAIL_CONFIGURED:
        # Don't show error here, just log and return false.
        # The calling function (auth_component) will show a warning.
        print("Email notification skipped: Configuration incomplete.")
        return False

    subject = f"Successful Login to {APP_NAME}"
    body = f"""
    <html>
        <body>
            <p>Hello,</p>
            <p>You have successfully logged in to your {APP_NAME} account at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.</p>
            <p>If this was not you, please contact us immediately or secure your account.</p>
            <p>Thank you,<br>{APP_NAME} Team</p>
        </body>
    </html>
    """
    return send_email(email, subject, body, is_html=True)

def send_signup_notification(email):
    """Send welcome email after successful signup."""
    # Check configuration *before* proceeding
    if not EMAIL_CONFIGURED:
        print("Welcome email skipped: Configuration incomplete.")
        return False

    subject = f"Welcome to {APP_NAME}!"
    body = f"""
    <html>
        <body>
            <p>Hello,</p>
            <p>Thank you for creating an account with {APP_NAME}!</p>
            <p>We're excited to help you discover personalized news content tailored to your interests.</p>
            <p>Get started by exploring different news categories and saving your favorite articles.</p>
            <p>If you have any questions, feel free to reply to this email (if configured).</p>
            <p>Happy reading!<br>{APP_NAME} Team</p>
        </body>
    </html>
    """
    return send_email(email, subject, body, is_html=True)


# --- Helper Functions ---

def format_date(date_str):
    """Formats ISO date string to a readable format, handling timezones."""
    if not date_str:
        return "Date not available"
    try:
        # Handle potential 'Z' for UTC
        dt_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        # Convert to local timezone for display (optional, requires tzlocal library or careful handling)
        # For simplicity, display in its original timezone offset or UTC if 'Z' was present
        return dt_obj.strftime("%b %d, %Y %I:%M %p %Z") # Include timezone info
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not parse date '{date_str}'. Error: {e}")
        # Fallback to just showing the string if parsing fails
        return str(date_str)

def highlight_text(text, search_terms):
    """Highlight search terms in text (case-insensitive)"""
    if not text or not search_terms:
        return text

    highlighted_text = str(text) # Ensure text is a string
    # Filter out empty strings from search terms
    valid_terms = [term for term in search_terms if term and term.strip()]
    if not valid_terms:
        return highlighted_text

    for term in valid_terms:
        try:
            # Escape special regex characters in the search term
            pattern = re.compile(re.escape(term.strip()), re.IGNORECASE)
            # Use a function for replacement to handle potential overlaps slightly better
            # and ensure the original case inside the tag matches the found text
            highlighted_text = pattern.sub(
                lambda match: f'<mark style="background-color: #FFFF00; font-weight: bold;">{match.group(0)}</mark>',
                highlighted_text
            )
        except re.error as e:
            # Log regex errors but continue with other terms
            print(f"Regex error highlighting term '{term}': {e}")
            continue
        except Exception as e:
             print(f"Error during highlighting term '{term}': {e}")
             continue # Skip this term if any other error occurs
    return highlighted_text


# --- API Interaction ---

@st.cache_data(ttl=600) # Cache for 10 minutes
def fetch_news(query, page_size=DEFAULT_PAGE_SIZE, from_date=None, to_date=None, sort_by="relevancy"):
    """Fetches news from NewsAPI /everything endpoint."""
    if not API_KEY:
        st.error("NewsAPI Key is not configured.")
        return []
    if not query or not query.strip():
         # Allow empty query internally for Saved/Liked, but warn if user input is empty
         # st.warning("Search query cannot be empty for fetching news.")
         # For Saved/Liked, a broad query is used later, so allow it here.
         # If called directly with empty query from user input, it should be caught before calling.
         print("fetch_news called with empty or whitespace query.") # Log internal calls
         return []

    params = {
        'q': query.strip(), # Ensure no leading/trailing whitespace
        'apiKey': API_KEY,
        'pageSize': min(page_size, 100), # NewsAPI max page size is 100
        'sortBy': sort_by,
        "language": "en" # Focus on English news
    }
    # Add date parameters if provided
    if from_date:
        params['from'] = from_date
    if to_date:
        params['to'] = to_date

    # Ensure query is not excessively long (NewsAPI might have limits)
    if len(params['q']) > 500:
        st.warning("Search query is very long, potentially exceeding API limits. Truncating.")
        print(f"Query truncated: {params['q'][:500]}...")
        params['q'] = params['q'][:500]

    try:
        response = requests.get(BASE_URL, params=params, timeout=15) # 15-second timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # Check for API-specific errors reported in the JSON response
        if data.get('status') == 'error':
             st.error(f"NewsAPI Error ({data.get('code')}): {data.get('message', 'Unknown error')}")
             return []
        # Filter out articles with '[Removed]' title or description, as they are often useless
        articles = [
             article for article in data.get('articles', [])
             if article.get('title') != '[Removed]' and article.get('url')
         ]
        return articles

    except requests.exceptions.Timeout:
        st.error("Error fetching news: The request to NewsAPI timed out.")
        return []
    except requests.exceptions.HTTPError as e:
         st.error(f"HTTP Error fetching news: {e.response.status_code} {e.response.reason}")
         # Log the response body for debugging if possible
         try:
              print(f"NewsAPI HTTP Error Response: {e.response.json()}")
         except:
              print(f"NewsAPI HTTP Error Response (non-JSON): {e.response.text}")
         return []
    except requests.exceptions.RequestException as e:
        # Handle other potential network errors (DNS, connection, etc.)
        st.error(f"Error fetching news: A network error occurred: {str(e)}")
        return []
    except Exception as e:
        # Catch any other unexpected errors during the process
        st.error(f"An unexpected error occurred during news fetch: {e}")
        import traceback
        traceback.print_exc() # Log stack trace for debugging
        return []


@st.cache_data(ttl=120) # Cache for 2 minutes for headlines
def fetch_top_headlines(category=None, country="us", page_size=1):
    """Fetches top headlines for a specific category or general headlines."""
    if not API_KEY:
        print("Error: NewsAPI Key is not configured for top headlines.")
        # Return None instead of erroring in UI, as this might be optional
        return None

    params = {
        'apiKey': API_KEY,
        'pageSize': min(page_size, 100),
        'country': country,
    }
    # Add category only if it's provided and valid for the endpoint
    if category and category.lower() in NEWSAPI_TOP_HEADLINE_CATEGORIES:
        params['category'] = category.lower()
    elif category:
         # Log if an invalid category was attempted (e.g., from preferences)
         print(f"Warning: Category '{category}' is not valid for NewsAPI top-headlines. Fetching general headlines.")

    try:
        response = requests.get(HEADLINES_URL, params=params, timeout=10) # 10-second timeout for headlines
        response.raise_for_status()
        data = response.json()

        if data.get('status') == 'error':
             # Log API errors instead of showing in UI directly for this component
             print(f"NewsAPI Top Headlines Error ({data.get('code')}): {data.get('message', 'Unknown error')}")
             return None

        articles = data.get('articles', [])
        # Filter out removed articles
        valid_articles = [
             article for article in articles
             if article.get('title') != '[Removed]' and article.get('url')
         ]

        # Return the first valid article if page_size=1, otherwise return list
        if page_size == 1:
             return valid_articles[0] if valid_articles else None
        else:
             return valid_articles # Return the list of articles

    except requests.exceptions.Timeout:
        print(f"Error fetching top headlines for category '{category}': Request timed out.")
        return None
    except requests.exceptions.HTTPError as e:
         print(f"HTTP Error fetching top headlines for category '{category}': {e.response.status_code} {e.response.reason}")
         return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching top headlines for category '{category}': {str(e)}")
        return None
    except Exception as e:
        # Catch unexpected errors
        print(f"An unexpected error occurred fetching headlines for category '{category}': {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# --- Firebase Interaction ---

def track_user_activity(user_id, article, action_type="view"):
    """Track user activity in Firestore"""
    # Basic validation
    if not user_id or not article or not isinstance(article, dict) or not article.get('url'):
        print("Warning: Invalid data passed to track_user_activity.")
        return

    valid_actions = ["view", "save", "like", "dislike"]
    if action_type not in valid_actions:
        print(f"Warning: Invalid action type '{action_type}' passed to track_user_activity.")
        return

    try:
        activity_ref = db.collection("users").document(user_id).collection("activity")
        # Prepare data, ensuring values are present or None
        activity_data = {
            "article_title": article.get('title'),
            "article_description": article.get('description'),
            "article_url": article.get('url'), # Should always be present based on validation
            "article_source": article.get('source', {}).get('name'),
            "published_at": article.get('publishedAt'),
            # Get category/subcategory from session state if available
            "category": st.session_state.get('current_category'),
            "subcategory": st.session_state.get('current_subcategory'),
            "action_type": action_type,
            "timestamp": firestore.SERVER_TIMESTAMP # Use server timestamp
        }
        # Remove keys with None values if desired, or Firestore handles them
        # activity_data = {k: v for k, v in activity_data.items() if v is not None}

        activity_ref.add(activity_data)

    except exceptions.FirebaseError as e:
        st.error(f"Error tracking activity (Firebase): {str(e)}")
        print(f"Firestore error during activity tracking for user {user_id}: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred tracking activity: {str(e)}")
        print(f"Unexpected error during activity tracking for user {user_id}: {e}")
        import traceback
        traceback.print_exc()


def get_user_preferences(user_id):
    """Get user preferences from Firestore"""
    if not user_id: return {}
    try:
        doc_ref = db.collection("users").document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            user_data = doc.to_dict()
            prefs = user_data.get('preferences', {})
            # Ensure default structure if preferences exist but keys are missing
            if 'categories' not in prefs:
                 # Default to all non-internal categories if missing
                 prefs['categories'] = list(cat for cat in CATEGORIES if not cat.startswith('_'))
            if 'notifications' not in prefs:
                 prefs['notifications'] = False # Default to false
            return prefs
        else:
             # Return default preferences if the user document doesn't exist (shouldn't happen post-signup)
             print(f"Warning: User document not found for user_id {user_id} in get_user_preferences.")
             return {
                 'categories': list(cat for cat in CATEGORIES if not cat.startswith('_')),
                 'notifications': False
             }
    except exceptions.FirebaseError as e:
        st.error(f"Error getting preferences (Firebase): {str(e)}")
        print(f"Firestore error getting preferences for user {user_id}: {e}")
        return {} # Return empty dict on error to avoid breaking UI
    except Exception as e:
        st.error(f"An unexpected error occurred getting preferences: {str(e)}")
        print(f"Unexpected error getting preferences for user {user_id}: {e}")
        return {}

def update_user_preferences(user_id, preferences):
    """Update user preferences in Firestore"""
    if not user_id or not isinstance(preferences, dict):
        print("Warning: Invalid data passed to update_user_preferences.")
        return
    try:
        doc_ref = db.collection("users").document(user_id)
        # Use merge=True to only update the 'preferences' field and add 'last_updated'
        doc_ref.set({
            'preferences': preferences,
            'last_updated': firestore.SERVER_TIMESTAMP
        }, merge=True)
    except exceptions.FirebaseError as e:
        st.error(f"Error updating preferences (Firebase): {str(e)}")
        print(f"Firestore error updating preferences for user {user_id}: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred updating preferences: {str(e)}")
        print(f"Unexpected error updating preferences for user {user_id}: {e}")


def get_user_activity(user_id, limit=50):
    """Get user activity from Firestore, ordered by time descending."""
    if not user_id: return []
    try:
        activity_ref = (db.collection("users").document(user_id)
                        .collection("activity")
                        .order_by("timestamp", direction=firestore.Query.DESCENDING)
                        .limit(limit))
        docs = activity_ref.stream()
        # Convert documents to dictionaries
        return [doc.to_dict() for doc in docs]
    except exceptions.FirebaseError as e:
        st.error(f"Error getting activity (Firebase): {str(e)}")
        print(f"Firestore error getting activity for user {user_id}: {e}")
        return [] # Return empty list on error
    except Exception as e:
        st.error(f"An unexpected error occurred getting activity: {str(e)}")
        print(f"Unexpected error getting activity for user {user_id}: {e}")
        return []


# --- Personalization / Recommendation ---

def get_personalized_recommendations(user_id, all_articles, activity_limit=100, min_recommendations=5):
    """
    Simple keyword-based recommendation engine based on user activity (like, save, view, dislike).
    Boosts recent articles. Returns a ranked list of articles.
    """
    if not user_id or not all_articles:
        return all_articles # Return original list if no user or articles

    user_activity = get_user_activity(user_id, limit=activity_limit)
    if not user_activity:
        # No activity, return original list (maybe sorted by date?)
        all_articles.sort(key=lambda x: x.get('publishedAt', ''), reverse=True)
        return all_articles[:DEFAULT_PAGE_SIZE] # Limit to default page size

    # --- Keyword Extraction from Activity ---
    # Using sets for efficient storage and lookup
    liked_keywords = set()
    saved_keywords = set()
    viewed_keywords = set()
    disliked_keywords = set()

    # Simple regex for potentially meaningful words (adjust length as needed)
    keyword_regex = re.compile(r'\b\w{4,}\b') # Words with 4+ letters

    for activity in user_activity:
        # Combine title and description for keyword source
        text_content = f"{activity.get('article_title', '') or ''} {activity.get('article_description', '') or ''}".lower()
        if not text_content.strip(): continue # Skip if no text

        keywords = set(keyword_regex.findall(text_content))
        if not keywords: continue # Skip if no keywords found

        action = activity.get('action_type')
        if action == 'like':
            liked_keywords.update(keywords)
        elif action == 'save':
            saved_keywords.update(keywords)
        elif action == 'dislike':
             disliked_keywords.update(keywords)
        elif action == 'view':
            # Maybe give lower weight to viewed keywords if they weren't liked/saved
            viewed_keywords.update(keywords)

    # Remove disliked keywords from liked/saved/viewed sets for stronger negative signal
    liked_keywords -= disliked_keywords
    saved_keywords -= disliked_keywords
    viewed_keywords -= disliked_keywords

    # --- Scoring Articles ---
    scored_articles = []
    now_utc = datetime.now(timezone.utc)

    for article in all_articles:
        score = 0
        article_text_lower = f"{article.get('title', '') or ''} {article.get('description', '') or ''}".lower()
        if not article_text_lower.strip(): continue # Skip articles with no text content

        article_keywords = set(keyword_regex.findall(article_text_lower))
        if not article_keywords: continue # Skip articles with no keywords

        # Score based on keyword overlap with activity types (adjust weights as needed)
        score += len(article_keywords.intersection(liked_keywords)) * 5   # High weight for likes
        score += len(article_keywords.intersection(saved_keywords)) * 3   # Medium weight for saves
        score += len(article_keywords.intersection(viewed_keywords)) * 1  # Low weight for views
        # Negative score for disliked keywords already handled by removing them from positive sets

        # Boost score for recent articles (e.g., published within last 3 days)
        try:
             published_dt = datetime.fromisoformat(article.get('publishedAt','').replace('Z', '+00:00'))
             time_delta = now_utc - published_dt
             if time_delta < timedelta(days=1):
                 score += 3 # Big boost for last 24 hours
             elif time_delta < timedelta(days=3):
                 score += 1.5 # Smaller boost for last 3 days
             # Optional: Decay score slightly for older articles beyond a week?
             # elif time_delta > timedelta(days=7):
             #    score *= 0.95

        except (ValueError, TypeError):
             pass # Ignore articles with invalid dates for time boosting

        # Add the scored article (score, article dict) to the list
        scored_articles.append((score, article))

    # --- Ranking and Filtering ---
    # Sort articles by score in descending order
    scored_articles.sort(reverse=True, key=lambda x: x[0])

    # Filter out articles with non-positive scores (unless we need to fill)
    # A score of 0 means no positive keyword matches and not recent, likely irrelevant
    recommended_articles = [article for score, article in scored_articles if score > 0]

    # --- Fill with remaining articles if recommendation count is low ---
    if len(recommended_articles) < min_recommendations:
         needed = min_recommendations - len(recommended_articles)
         # Get URLs of already recommended articles
         recommended_urls = {a['url'] for a in recommended_articles}
         # Get remaining articles (those not recommended, including score <= 0)
         # Sort remaining articles by published date as a fallback ranking
         remaining_articles = [article for score, article in scored_articles if article['url'] not in recommended_urls]
         remaining_articles.sort(key=lambda x: x.get('publishedAt', ''), reverse=True)

         # Add the needed number of remaining articles
         recommended_articles.extend(remaining_articles[:needed])

    # Return the top N recommendations, limited by DEFAULT_PAGE_SIZE
    return recommended_articles[:DEFAULT_PAGE_SIZE]


# --- UI Components ---

def display_article(article, user_id=None, search_terms=None):
    """Displays a single article card with interaction buttons."""
    # Basic validation for the article object
    if not article or not isinstance(article, dict) or not article.get('url'):
         # Don't display anything if the article data is invalid/missing URL
         return

    # Use URL as a base for unique keys for buttons within this article display
    # Using hash() for potentially shorter keys if needed, but URL is safer for uniqueness
    unique_key_base = article['url']

    # Use st.container to group elements for better layout control
    with st.container():
        # Use columns for image + text layout
        col1, col2 = st.columns([1, 3], gap="medium") # Adjust ratio if needed (e.g., [1, 4])

        with col1:
            # Display image, use placeholder if urlToImage is missing or empty
            image_url = article.get('urlToImage')
            if not image_url:
                 image_url = PLACEHOLDER_IMAGE
            st.image(
                image_url,
                use_container_width=True,
                # Add caption with source name if available
                caption=f"Source: {article.get('source', {}).get('name', 'Unknown')}"
            )

        with col2:
            title = article.get('title', 'No Title Provided')
            description_raw = article.get('description') # Get potential None
            description = description_raw if description_raw else 'No description available.' # Ensure string


            # Apply highlighting if search terms are provided
            if search_terms:
                 display_title = highlight_text(title, search_terms)
                 display_desc_highlighted = highlight_text(description, search_terms)
                 # Display title using markdown for highlighting
                 st.markdown(f"<h4>{display_title}</h4>", unsafe_allow_html=True)
            else:
                 # Display title using st.subheader for standard styling
                 st.subheader(title)
                 display_desc_highlighted = description # No highlighting needed

            # Display metadata (Author, Date) concisely
            metadata = []
            author = article.get('author')
            if author:
                # Truncate long author lists/strings if necessary
                metadata.append(f"By {author[:50]}{'...' if len(author)>50 else ''}")
            published_at = format_date(article.get('publishedAt'))
            metadata.append(published_at)
            st.caption(" • ".join(filter(None, metadata))) # Filter ensures no extra separators if item is missing

            # Display description using markdown (allows highlighting and basic formatting)
            # Truncate long descriptions
            max_desc_len = 250 # Adjust as needed
            if len(display_desc_highlighted) > max_desc_len:
                 # Check if highlighting exists before deciding where to cut? Simpler just to cut.
                 display_desc_truncated = display_desc_highlighted[:max_desc_len] + '...'
            else:
                 display_desc_truncated = display_desc_highlighted

            st.markdown(f"<p>{display_desc_truncated}</p>", unsafe_allow_html=True)

            # Link to the full article
            st.markdown(f"[Read full article]({article['url']})", unsafe_allow_html=True)

            # Interaction buttons (only if user is logged in)
            if user_id:
                # Use columns for button layout
                btn_col1, btn_col2, btn_col3, spacer = st.columns([1, 1, 1, 3]) # Adjust spacing if needed
                with btn_col1:
                    if st.button("👍 Like", key=f"like_{unique_key_base}", help="Like this article"):
                        track_user_activity(user_id, article, "like")
                        st.toast("Liked!", icon="👍")
                        # Optional: Rerun to update recommendations immediately? Could be disruptive.
                        # time.sleep(0.5) # Short delay for toast
                        # st.rerun()
                with btn_col2:
                    if st.button("👎 Dislike", key=f"dislike_{unique_key_base}", help="Dislike this article"):
                        track_user_activity(user_id, article, "dislike")
                        st.toast("Feedback noted!", icon="👎")
                        # Optional: Rerun?
                        # time.sleep(0.5)
                        # st.rerun()
                with btn_col3:
                    if st.button("💾 Save", key=f"save_{unique_key_base}", help="Save for later"):
                        track_user_activity(user_id, article, "save")
                        st.toast("Article Saved!", icon="💾")
                        # Saving doesn't usually require immediate rerun
            else:
                 # Subtle hint for non-logged-in users
                 st.caption("Login or Sign Up to like, dislike, or save articles.")

        # Track view implicitly when displayed (if user is logged in)
        # Use session state to track views *per session* to avoid repeated tracking on reruns
        if user_id:
             if 'view_tracked' not in st.session_state:
                 st.session_state.view_tracked = set()
             if article['url'] not in st.session_state.view_tracked:
                 track_user_activity(user_id, article, "view")
                 st.session_state.view_tracked.add(article['url'])

        # Add a visual separator between articles
        st.markdown("---")


def auth_component():
    """Handles user authentication (Login/Signup) in the sidebar with email notifications."""
    # Initialize user state if not present
    if 'user' not in st.session_state:
        st.session_state.user = None

    with st.sidebar:
        st.title("Account")

        # --- Display logged-in state ---
        if st.session_state.user:
            st.write(f"Welcome,")
            st.write(f"**{st.session_state.user['email']}**") # Make email bold
            if st.button("Sign Out"):
                user_email_on_logout = st.session_state.user['email'] # Get email before clearing
                st.session_state.user = None
                # Clear potentially sensitive session state items on logout
                st.session_state.pop('view_tracked', None)
                st.session_state.pop('current_notification_article', None)
                st.session_state.pop('last_notification_check', None)
                st.toast(f"Signed out {user_email_on_logout}")
                st.rerun()

        # --- Display login/signup form ---
        else:
            choice = st.selectbox("Login / Sign Up", ["Login", "Sign Up"], label_visibility="collapsed")

            with st.form("auth_form", clear_on_submit=False): # Keep values on failed attempt
                email = st.text_input("Email Address", key="auth_email")
                password = st.text_input("Password", type="password", key="auth_password")
                submitted = st.form_submit_button(choice)

                if submitted:
                    # Basic client-side validation
                    if not email or not password:
                         st.error("Email and Password are required.")
                         return # Stop processing if fields are empty

                    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                         st.error("Please enter a valid email address.")
                         return

                    # --- Login Logic ---
                    if choice == "Login":
                        try:
                            # Verify user exists via Firebase Admin SDK (doesn't verify password)
                            user = auth.get_user_by_email(email)

                            # !! IMPORTANT SECURITY NOTE !!
                            # Verifying password requires client-side Firebase Auth SDK (JS or mobile).
                            # This backend-only check only confirms the email exists.
                            # For a real app, implement proper password verification on the client
                            # or use a more secure backend method if available (e.g., custom tokens).
                            st.warning("DEMO LOGIN: Email found. Password not verified (requires client-side SDK).")

                            st.session_state.user = {
                                'email': user.email,
                                'uid': user.uid
                            }

                            # Send login notification email (if configured)
                            if EMAIL_CONFIGURED:
                                if send_login_notification(user.email):
                                    st.toast(f"Login successful! Notification sent to {user.email}", icon="✉️")
                                else:
                                    # Error is shown by send_email, maybe add context here
                                    st.warning("Login successful, but failed to send notification email.")
                            else:
                                st.warning("Login successful! (Email notifications are not configured)")

                            st.rerun() # Rerun to update the main app view

                        except exceptions.NotFoundError:
                            st.error("Login Failed: User not found. Please check your email or sign up.")
                        except exceptions.FirebaseError as e:
                            st.error(f"Login Failed: An error occurred during authentication. ({str(e)})")
                            print(f"Firebase login error for {email}: {e}")
                        except Exception as e:
                            st.error(f"Login Failed: An unexpected error occurred. {str(e)}")
                            print(f"Unexpected login error for {email}: {e}")


                    # --- Sign Up Logic ---
                    elif choice == "Sign Up":
                        if len(password) < 6:
                            st.error("Sign Up Failed: Password must be at least 6 characters long.")
                            return

                        try:
                            # Create user using Firebase Admin SDK
                            user = auth.create_user(
                                email=email,
                                password=password,
                                email_verified=False # User needs to verify email separately if needed
                            )
                            st.success(f"Account created for {user.email}!")
                            st.info("Please log in using your new credentials.") # Ask user to login explicitly

                            # Initialize user data in Firestore immediately after creation
                            try:
                                default_prefs = {
                                    # Default to all non-internal categories
                                    'categories': list(cat for cat in CATEGORIES if not cat.startswith('_')),
                                    'notifications': False # Default notifications off
                                }
                                db.collection("users").document(user.uid).set({
                                    'email': email,
                                    'preferences': default_prefs,
                                    'created_at': firestore.SERVER_TIMESTAMP
                                })
                                print(f"Firestore profile created for new user {user.uid}")
                            except Exception as db_error:
                                # Log error creating profile, but user account exists
                                print(f"Error creating Firestore profile for {user.uid} after signup: {db_error}")
                                st.warning("Account created, but failed to initialize profile data.")


                            # Send welcome email (if configured)
                            if EMAIL_CONFIGURED:
                                if send_signup_notification(email):
                                    st.toast("Welcome email sent to your inbox!", icon="✉️")
                                else:
                                    st.warning("Account created, but failed to send welcome email.")
                            else:
                                st.warning("Account created! (Email notifications are not configured)")

                            # Do NOT automatically log in the user after signup here
                            # Force them to go through the login flow again.
                            # Clear the form fields after successful signup attempt
                            # st.session_state.auth_email = "" # This doesn't work with st.form easily
                            # st.session_state.auth_password = ""

                        except exceptions.EmailAlreadyExistsError:
                            st.error("Sign Up Failed: This email address is already registered. Please log in instead.")
                        except exceptions.InvalidArgumentError:
                             st.error("Sign Up Failed: Invalid email address format.")
                        except exceptions.FirebaseError as e:
                            st.error(f"Sign Up Failed: An error occurred. ({str(e)})")
                            print(f"Firebase signup error for {email}: {e}")
                        except Exception as e:
                            st.error(f"Sign Up Failed: An unexpected error occurred. {str(e)}")
                            print(f"Unexpected signup error for {email}: {e}")


def preferences_component(user_id):
    """Component for user to set preferences in the sidebar."""
    if not user_id: return # Only show if logged in

    with st.sidebar:
        st.markdown("---")
        st.subheader("Your Preferences")

        # Use an expander to keep the sidebar cleaner initially
        with st.expander("Manage Preferences", expanded=False):
            # Fetch current preferences right before displaying
            # Avoid caching this directly as it needs to be fresh
            preferences = get_user_preferences(user_id)

            # --- Category Preferences ---
            # Get all available category keys (excluding internal ones like _personalized_)
            all_category_keys = [key for key in CATEGORIES if not key.startswith('_')]
            # Create a mapping from key to display name for the multiselect options
            available_display_categories_map = {
                key: CATEGORIES[key]['display_name'] for key in all_category_keys
            }
            # Get the keys of the currently preferred categories from Firestore data
            current_pref_keys = preferences.get('categories', all_category_keys) # Default to all if not set

            # Get the display names corresponding to the current preference keys
            # Ensure we only include valid/existing categories
            current_pref_display = [
                available_display_categories_map[key]
                for key in current_pref_keys if key in available_display_categories_map
            ]

            # Multiselect widget using display names
            preferred_display_categories = st.multiselect(
                "My News Categories",
                options=list(available_display_categories_map.values()), # Show display names
                default=current_pref_display, # Select current preferences by display name
                key="pref_categories"
            )

            # Convert selected display names back to their corresponding keys for saving
            # Create a reverse map from display name to key
            display_name_to_key_map = {v: k for k, v in available_display_categories_map.items()}
            preferred_keys = [
                display_name_to_key_map[display_name]
                for display_name in preferred_display_categories
                if display_name in display_name_to_key_map # Ensure valid selection
            ]


            # --- Notification Preferences ---
            # Checkbox for enabling/disabling the breaking news notification panel
            notifications_enabled = st.checkbox(
                "Show breaking news panel",
                value=preferences.get('notifications', False), # Default false if not set
                key="pref_notifications",
                help="Enable a panel showing the latest headline from your preferred categories."
            )

            # --- Save Button ---
            if st.button("Save Preferences"):
                # Prepare the updated preferences dictionary
                new_preferences = {
                    'categories': preferred_keys,
                    'notifications': notifications_enabled
                }
                # Call Firestore update function
                update_user_preferences(user_id, new_preferences)
                st.success("Preferences saved successfully!")
                # Clear relevant caches that might depend on preferences
                st.cache_data.clear() # Clears all @st.cache_data
                # Optional: Clear specific caches if identifiable
                # fetch_news.clear()
                # fetch_top_headlines.clear()
                # Rerun the script to reflect changes immediately in the UI
                time.sleep(1) # Short delay for user to see success message
                st.rerun()


def notification_panel_component(user_id):
    """ Displays a notification panel that checks for breaking news periodically. """
    if not user_id: return # Only for logged-in users

    # Check user preference first - don't proceed if disabled
    prefs = get_user_preferences(user_id)
    if not prefs.get('notifications', False):
        return # Exit silently if notifications are disabled in preferences

    # --- Configuration ---
    CHECK_INTERVAL_SECONDS = 180 # Check every 3 minutes (adjust as needed)

    # --- Session State Initialization ---
    # Store the last check time and the currently displayed article URL to avoid flickering
    if 'last_notification_check' not in st.session_state:
        # Initialize to 0 to force an immediate check on first load
        st.session_state.last_notification_check = 0
    if 'current_notification_article' not in st.session_state:
        # Store the actual article dictionary
        st.session_state.current_notification_article = None

    current_time = time.time()

    # --- Check if Interval Has Passed ---
    if current_time - st.session_state.last_notification_check > CHECK_INTERVAL_SECONDS:
        # print("DEBUG: Checking for new notification...") # Optional server log
        st.session_state.last_notification_check = current_time

        # Fetch the *single* latest breaking news article relevant to the user
        latest_breaking_news = get_latest_breaking_news_for_user(user_id)

        # --- Update Notification State ---
        # Get URL of the currently displayed notification (if any)
        current_displayed_url = None
        if st.session_state.current_notification_article:
            current_displayed_url = st.session_state.current_notification_article.get('url')

        # Get URL of the newly fetched notification (if any)
        new_fetched_url = None
        if latest_breaking_news:
            new_fetched_url = latest_breaking_news.get('url')

        # Update the session state *only* if a new, different article is found
        if new_fetched_url and new_fetched_url != current_displayed_url:
            st.session_state.current_notification_article = latest_breaking_news
            # Force a rerun to display the new notification immediately
            # print(f"DEBUG: New notification found: {new_fetched_url}") # Optional log
            st.rerun()
        elif not latest_breaking_news and st.session_state.current_notification_article:
            # Optional: Decide whether to clear the notification if no new news is found
            # For now, keep displaying the last found one until a new one replaces it
             # print("DEBUG: No new breaking news found, keeping existing notification.")
             pass


    # --- Display the Current Notification ---
    # Use the article stored in session state
    notification_to_display = st.session_state.current_notification_article
    if notification_to_display:
        with st.container():
            # --- CSS Styling for the Panel ---
            st.markdown(
                """
                <style>
                .notification-panel {
                    border: 1px solid #dee2e6; /* Light grey border */
                    border-left: 5px solid #17a2b8; /* Info blue left border */
                    border-radius: 0.375rem; /* Standard border radius */
                    padding: 0.8rem 1rem; /* Adjust padding */
                    background-color: #f0f8ff; /* Alice blue - very light */
                    margin-bottom: 1.5rem; /* Space below the panel */
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1); /* Subtle shadow */
                    transition: background-color 0.3s ease; /* Smooth transition effect */
                }
                /* Optional: slight hover effect */
                /* .notification-panel:hover { background-color: #e6f2ff; } */

                .notification-panel p { margin-bottom: 0.25rem; font-size: 0.95em;} /* Smaller text */
                .notification-title a {
                    color: #0056b3 !important; /* Darker blue link */
                    font-weight: 600; /* Semi-bold */
                    text-decoration: none; /* No underline by default */
                }
                .notification-title a:hover {
                    text-decoration: underline; /* Underline on hover */
                    color: #003d80 !important; /* Even darker blue on hover */
                }
                .notification-caption { font-size: 0.8em; color: #6c757d; } /* Smaller, grey caption */
                </style>
                """, unsafe_allow_html=True
            )

            # --- Panel Content ---
            # Apply the CSS class to a markdown container
            st.markdown('<div class="notification-panel">', unsafe_allow_html=True)

            title = notification_to_display.get('title', 'No Title')
            url = notification_to_display.get('url', '#') # Fallback URL
            source = notification_to_display.get('source', {}).get('name', 'Unknown Source')
            published_at = format_date(notification_to_display.get('publishedAt'))

            # Display Title with Link
            st.markdown(f"<p class='notification-title'>🔔  <b>Breaking:</b> <a href='{url}' target='_blank' title='{title}'>{title}</a></p>", unsafe_allow_html=True)
            # Display Caption
            st.markdown(f"<p class='notification-caption'>Source: {source} | Published: {published_at}</p>", unsafe_allow_html=True)

            # Close the div
            st.markdown('</div>', unsafe_allow_html=True)


def get_latest_breaking_news_for_user(user_id):
    """ Gets the single latest breaking news article across user's preferred valid NewsAPI categories. """
    if not user_id: return None

    preferences = get_user_preferences(user_id)
    # Get category *keys* from user preferences
    preferred_keys = preferences.get('categories', [])
    if not preferred_keys:
         # print("DEBUG: User has no preferred categories set.")
         return None # No categories to check

    # Filter preference keys to only include those valid for NewsAPI's /top-headlines 'category' parameter
    categories_to_check = [key for key in preferred_keys if key in NEWSAPI_TOP_HEADLINE_CATEGORIES]

    if not categories_to_check:
        # print(f"DEBUG: User's preferred categories ({preferred_keys}) contain no valid NewsAPI top-headline categories ({NEWSAPI_TOP_HEADLINE_CATEGORIES}).")
        return None # No valid categories to check

    latest_article = None
    # Initialize with timezone-aware minimum datetime for proper comparison
    latest_timestamp_utc = datetime.min.replace(tzinfo=timezone.utc)

    # Iterate through the valid preferred categories
    for category_key in categories_to_check:
        # Fetch only the *single latest* headline for this category
        # fetch_top_headlines returns a single article dict or None when page_size=1
        article = fetch_top_headlines(category=category_key, page_size=1, country="us") # Specify country if needed

        if article and article.get('publishedAt'):
            try:
                # Parse the published date string, assuming UTC ('Z' or offset)
                published_time_utc = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))

                # Compare with the latest timestamp found so far
                if published_time_utc > latest_timestamp_utc:
                    latest_timestamp_utc = published_time_utc
                    latest_article = article # Update the latest article

            except (ValueError, TypeError) as e:
                # Log errors during date parsing but continue checking other categories
                print(f"Error parsing date '{article.get('publishedAt')}' for notification article in category '{category_key}': {e}")
                continue
        # else: print(f"DEBUG: No article found for category {category_key}") # Optional log

    # if latest_article: print(f"DEBUG: Found latest article: {latest_article.get('title')}") # Optional log
    # else: print("DEBUG: No latest article found across preferred categories.") # Optional log

    return latest_article # Return the single latest article found, or None


# --- Main Application Structure ---

def display_regular_news_feed(user_id):
    """Displays the regular news feed tab content, driven by sidebar selections."""

    # --- Sidebar Setup ---
    sidebar_from_date, sidebar_to_date = None, None # Dates from sidebar filter
    selected_category_key = None
    selected_subcategory_name = None
    search_term_from_sidebar = None
    category_display_name = "News" # Default display name
    is_personalized_feed = False
    is_saved_liked_feed = False
    action_type_to_fetch = None # 'save' or 'like' for saved/liked feeds

    with st.sidebar:
        # --- Date Filtering ---
        st.subheader("Filter by Date")
        # Use more descriptive options
        date_filter_options = ["Recent (24h)", "Last 3 Days", "Last Week", "Last Month", "All Time"]
        date_filter = st.radio(
            "Time Period", date_filter_options, index=2, # Default 'Last Week'
            key="feed_date_filter", label_visibility="collapsed", horizontal=True
        )

        # Calculate date range based on selection (used for non-saved/liked feeds)
        now_utc = datetime.now(timezone.utc)
        if date_filter != "All Time":
            if date_filter == "Recent (24h)":
                sidebar_from_date = now_utc - timedelta(days=1)
            elif date_filter == "Last 3 Days":
                 sidebar_from_date = now_utc - timedelta(days=3)
            elif date_filter == "Last Week":
                sidebar_from_date = now_utc - timedelta(weeks=1)
            else: # Last Month
                sidebar_from_date = now_utc - timedelta(days=30) # Approx 1 month

            # Format dates as YYYY-MM-DD strings for the API
            sidebar_from_date = sidebar_from_date.strftime('%Y-%m-%d')
            sidebar_to_date = now_utc.strftime('%Y-%m-%d') # To date is always today for these ranges


        st.markdown("---")
        st.subheader("Categories")

        # --- Category Selection ---
        # Get all category keys (including internal ones if logged in)
        all_category_keys = list(CATEGORIES.keys())
        if not user_id: # Exclude personalized category if not logged in
            all_category_keys = [k for k in all_category_keys if not k.startswith('_')]

        # Get user's preferred category keys if logged in
        pref_keys = []
        if user_id:
            pref_keys = get_user_preferences(user_id).get('categories', [])

        # Determine categories to display in the radio button list
        # Show personalized if logged in, plus preferred categories, plus maybe 'general' as a fallback?
        available_cat_keys_display = []
        if user_id:
             available_cat_keys_display.append('_personalized_') # Always show personalized first if logged in
             # Add preferred keys (that are not internal)
             available_cat_keys_display.extend(sorted([key for key in pref_keys if not key.startswith('_')]))
             # Add 'general' if it wasn't in preferences? Optional.
             # if 'general' not in available_cat_keys_display:
             #      available_cat_keys_display.append('general')
        else:
             # Show all non-internal categories if not logged in
             available_cat_keys_display.extend(sorted([key for key in all_category_keys if not key.startswith('_')]))

        # Ensure no duplicates if 'general' was added manually
        available_cat_keys_display = list(dict.fromkeys(available_cat_keys_display)) # Remove duplicates while preserving order

        # Create map of key -> display name for the format_func
        cat_display_options = {key: CATEGORIES[key]['display_name'] for key in available_cat_keys_display if key in CATEGORIES}

        # Set default index: 0 (Personalized if logged in, first category otherwise)
        default_cat_index = 0

        selected_category_key = st.radio(
            "Select Category:",
            available_cat_keys_display,
            format_func=lambda key: cat_display_options.get(key, key.capitalize()), # Show display name, fallback to key
            index=default_cat_index,
            key="feed_category",
            label_visibility="collapsed"
        )
        category_display_name = cat_display_options.get(selected_category_key, selected_category_key.capitalize())
        # Store the *selected* category key in session state for activity tracking
        st.session_state.current_category = selected_category_key


        # --- Subcategory Selection ---
        # Check if the selected category has subcategories defined
        if selected_category_key in CATEGORIES and 'subcategories' in CATEGORIES[selected_category_key]:
            # st.markdown("---") # Optional separator
            st.subheader(f"{category_display_name} Topics") # Use display name
            subcategories_map = CATEGORIES[selected_category_key]["subcategories"]
            # Get the display names (keys of the subcategory map)
            subcategory_names = list(subcategories_map.keys())

            # Default index for subcategory (usually the first one)
            default_subcat_index = 0

            selected_subcategory_name = st.radio(
                f"Select Topic:",
                subcategory_names,
                index=default_subcat_index,
                key=f"feed_subcategory_{selected_category_key}", # Unique key per main category to reset selection
                label_visibility="collapsed"
            )
            # Store the *selected* subcategory display name
            st.session_state.current_subcategory = selected_subcategory_name

            # Determine the internal search term associated with the selected subcategory name
            search_term_from_sidebar = subcategories_map[selected_subcategory_name]

            # Handle special internal subcategories (_for_you_, _saved_, _liked_)
            if selected_category_key == '_personalized_':
                 is_personalized_feed = True
                 if search_term_from_sidebar == '_saved_':
                     is_saved_liked_feed = True
                     action_type_to_fetch = 'save'
                     category_display_name = "My Saved Articles"
                 elif search_term_from_sidebar == '_liked_':
                     is_saved_liked_feed = True
                     action_type_to_fetch = 'like'
                     category_display_name = "My Liked Articles"
                 else: # Default 'For You'
                      category_display_name = "News For You" # Override display name

                 # 'For You', 'Saved', 'Liked' don't use the subcategory value as a direct search term
                 if is_personalized_feed: search_term_from_sidebar = None

            else:
                 # For regular categories, update the display name to include the subcategory
                 category_display_name = f"{category_display_name}: {selected_subcategory_name}"

        else: # Category has no subcategories (or is invalid)
            # Use the main category key (or maybe its display name?) as the search term
            # Using the key is safer if display names have special characters
            search_term_from_sidebar = selected_category_key.lower() # Use lowercase key
            st.session_state.current_subcategory = None # No subcategory selected
            # Keep the main category display name


        # --- Quick Search Override ---
        st.markdown("---")
        st.subheader("Quick Search")
        custom_search = st.text_input(
            "🔍 Search all news:",
            key="quick_search",
            placeholder="Enter keywords...",
            label_visibility="collapsed",
            help="Overrides category selection. Uses selected date filter."
        ).strip() # Remove leading/trailing whitespace


        # --- Preferences Link/Component ---
        # (Placed here for logical flow in sidebar)
        preferences_component(user_id)


    # --- Determine Final Search Parameters & Fetching Logic ---
    final_search_term = None
    articles = []
    is_custom_search = bool(custom_search) # True if custom_search has content
    final_from_date = sidebar_from_date # Use dates calculated from sidebar radio
    final_to_date = sidebar_to_date
    highlight_terms = [] # Terms to highlight in results (only for custom search)
    category_color = "#6c757d" # Default color (grey)
    category_emoji = "📰" # Default emoji

    # --- Logic based on selection type ---
    if is_custom_search:
        final_search_term = custom_search
        category_display_name = f"Search Results: '{custom_search}'"
        category_color = "#6f42c1" # Purple for search?
        category_emoji = "🔍"
        highlight_terms = custom_search.split() # Simple split for highlighting
        # Uses sidebar date filter (final_from/to_date)

    elif is_saved_liked_feed:
        # Fetching logic is special (handled below), doesn't use final_search_term here
        # Uses fixed date range for retrieval, ignores sidebar date filter
        category_color = CATEGORIES['_personalized_']["color"]
        category_emoji = "💾" if action_type_to_fetch == 'save' else "👍"
        # Display name already set ('My Saved/Liked Articles')

    elif is_personalized_feed: # 'For You' feed
        # Fetching uses a broad query and recommendation logic (handled below)
        category_color = CATEGORIES['_personalized_']["color"]
        category_emoji = CATEGORIES['_personalized_']["emoji"]
        # Display name already set ('News For You')
        # Uses sidebar date filter for *candidate article* selection

    else: # Regular category/subcategory selected
         final_search_term = search_term_from_sidebar
         if selected_category_key in CATEGORIES:
            category_color = CATEGORIES[selected_category_key]["color"]
            category_emoji = CATEGORIES[selected_category_key]["emoji"]
         # else: # Use defaults if key somehow invalid (shouldn't happen)
         #     pass
         # Display name already set (includes subcategory if applicable)
         # Uses sidebar date filter (final_from/to_date)


    # --- Display Category Banner ---
    st.markdown( # Use markdown with inline CSS for the banner
        f"""<div style="
                background: linear-gradient(to right, {category_color}, {category_color}dd);
                color: white;
                padding: 0.8rem 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1.5rem;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
             <h2 style="color: white; margin: 0; font-weight: bold; font-size: 1.5em;">
                {category_emoji}   {category_display_name}
             </h2>
            </div>""",
        unsafe_allow_html=True
    )

    # --- Refresh Button ---
    # Place it near the top, maybe next to the banner? Or just below.
    if st.button("🔄 Refresh Feed", key="feed_refresh_top", help="Reload articles for the current selection"):
        st.cache_data.clear() # Clear relevant caches on refresh
        # Optionally clear specific caches if needed: fetch_news.clear(), etc.
        st.rerun()


    # --- Fetch News Articles ---
    # Use a spinner while fetching
    with st.spinner(f"Loading {category_display_name}..."):
        try: # Wrap fetching logic in try/except for robustness
            if is_saved_liked_feed:
                # --- Fetching Saved/Liked Articles ---
                activity = get_user_activity(user_id, limit=200) # Get recent activity
                # Extract unique URLs for the target action type
                target_urls = {
                    a['article_url'] for a in activity
                    if a.get('action_type') == action_type_to_fetch and a.get('article_url')
                }

                if not target_urls:
                    articles = [] # No saved/liked items found in activity
                else:
                    # Fetch candidate articles from the last 30 days (NewsAPI limit for /everything)
                    # Note: Free tier might only allow 24h lookback? Paid needed for 30 days.
                    # Use a very broad query, as we filter by URL anyway. Sort by date.
                    fetch_end_date = datetime.now(timezone.utc)
                    fetch_start_date = (fetch_end_date - timedelta(days=29)).strftime('%Y-%m-%d') # Look back ~30 days
                    fetch_end_date_str = fetch_end_date.strftime('%Y-%m-%d')

                    # Broad query needed by NewsAPI, sorting by date helps find recent ones first
                    # Use common words, or target specific domains if known? '*' might work on paid plans.
                    # Let's try a simple common word query.
                    candidate_query = "news OR article OR report"
                    fetched_candidates = fetch_news(
                        query=candidate_query,
                        page_size=100, # Fetch max candidates
                        from_date=fetch_start_date,
                        to_date=fetch_end_date_str,
                        sort_by="publishedAt"
                    )

                    # Filter the fetched candidates by the target URLs
                    articles_dict = {a['url']: a for a in fetched_candidates if a and a.get('url') in target_urls}
                    # Get the articles in the order of target_urls (which isn't time-ordered)
                    # or just use the articles found. Let's sort them by date again.
                    articles = sorted(articles_dict.values(), key=lambda x: x.get('publishedAt', ''), reverse=True)


                    # Provide feedback if some saved/liked articles weren't found
                    found_urls = {a['url'] for a in articles}
                    missing_count = len(target_urls - found_urls)
                    if missing_count > 0:
                        st.caption(f"ℹ️ Note: {missing_count} {action_type_to_fetch} article(s) could not be retrieved (may be older than ~30 days or removed by the source).")

            elif is_personalized_feed: # 'For You'
                # --- Fetching Personalized Recommendations ---
                # 1. Fetch a pool of candidate articles (broader query, recent dates)
                # Use a query based on user's preferred categories for better candidates?
                prefs = get_user_preferences(user_id)
                pref_keys_for_query = prefs.get('categories', ['general']) # Fallback to general
                # Join preferred keys with OR for a broad query
                candidate_query = " OR ".join(pref_keys_for_query) if pref_keys_for_query else "news" # Ensure query isn't empty

                candidate_articles = fetch_news(
                    query=candidate_query,
                    page_size=60, # Fetch a decent pool of candidates (adjust size as needed)
                    from_date=final_from_date, # Use sidebar date filter for candidates
                    to_date=final_to_date,
                    sort_by="popularity" # Fetch popular items to feed into personalization
                )
                # 2. Get recommendations based on the candidates and user activity
                articles = get_personalized_recommendations(user_id, candidate_articles)

            elif final_search_term: # Regular category or custom search
                # --- Fetching Standard News ---
                articles = fetch_news(
                    query=final_search_term,
                    page_size=DEFAULT_PAGE_SIZE, # Use default page size
                    from_date=final_from_date, # Use sidebar date filter
                    to_date=final_to_date,
                    # Sort search by relevance, category feeds by date (most recent first)
                    sort_by="relevancy" if is_custom_search else "publishedAt"
                )
            else:
                 # This case should ideally not be reached if sidebar logic is correct
                 st.warning("Please select a category or enter a search term.")
                 articles = []

        except Exception as e:
            st.error(f"An error occurred while fetching news: {e}")
            articles = [] # Ensure articles is empty on error


    # --- Display Articles ---
    if not articles:
        st.warning("No articles found matching your criteria.")
        # Provide specific guidance based on the context
        if is_saved_liked_feed:
            st.info(f"You haven't {action_type_to_fetch}d any articles recently, or they are older than ~30 days / no longer available.")
        elif is_personalized_feed:
            st.info("Your personalized feed refines over time. Interact with articles (like, save, view) to improve recommendations.")
        elif is_custom_search:
             st.info("Try broadening your search terms or adjusting the date filter.")
        else: # Regular category feed
             st.info("Try selecting a different topic, category, or adjusting the date filter.")
    else:
        # Show count of articles found/displayed
        st.success(f"Displaying {len(articles)} articles.")

        # Reset view tracking for this specific feed load/context
        # This ensures views are tracked correctly even if the user navigates back and forth
        # between feeds without a full page reload.
        st.session_state.view_tracked = set()

        # Iterate and display each article using the component function
        for article in articles:
            display_article(
                article,
                user_id,
                # Pass highlight terms only if it was a custom search
                search_terms=highlight_terms if is_custom_search else None
            )

        # Optional: Add a refresh button at the bottom as well
        if st.button("🔄 Refresh Feed", key="feed_refresh_bottom", help="Reload articles for the current selection"):
             st.cache_data.clear()
             st.rerun()


def display_advanced_search_interface(user_id):
    """Displays the advanced search tab content."""
    st.subheader("🔍 Advanced Article Search")
    st.write("Refine your search across all available news articles using more specific filters. Results are fetched from the last 30 days (NewsAPI limit).")

    with st.form(key="advanced_search_form"):
        # --- Search Query Input ---
        search_query = st.text_input(
            "**Keywords or Phrases**",
            placeholder="e.g., 'renewable energy policy' AND (europe OR germany)",
            key="adv_search_query",
            help="Enter your search terms. Use quotes for exact phrases, AND/OR/NOT for boolean logic (requires NewsAPI support, typically on paid plans for full boolean). Simple keyword matching works on all plans."
        )

        # --- Filters Row ---
        col1, col2 = st.columns(2)

        with col1:
            # --- Category Filter (Optional) ---
            st.markdown("**Filter by Categories (Optional)**")
            # Use non-internal categories for filtering options
            all_display_categories = [
                CATEGORIES[cat]['display_name'] for cat in CATEGORIES if not cat.startswith('_')
            ]
            selected_display_categories = st.multiselect(
                "Select categories to search within:",
                all_display_categories,
                key="adv_search_categories",
                label_visibility="collapsed",
                help="Results must match keywords AND belong to one of these categories (adds subcategory terms to query)."
            )
            # Convert selected display names back to keys for query building
            adv_cat_key_map = {CATEGORIES[key]['display_name']: key for key in CATEGORIES if not key.startswith('_')}
            selected_category_keys = [adv_cat_key_map[name] for name in selected_display_categories if name in adv_cat_key_map]

        with col2:
            # --- Date Range Filter ---
            st.markdown("**Time Period**")
            # Offer specific ranges relevant to NewsAPI free tier limits if needed
            # Free tier often limited to 24h on /everything
            # Paid tier up to 1 month (or more historically)
            # Let's assume paid tier access for month-long lookback here.
            date_range_options = ["Last 24 hours", "Last 7 days", "Last 30 days", "Custom range"]
            date_range = st.selectbox(
                "Select time period:", date_range_options, index=2, # Default 'Last 30 days'
                key="adv_search_date_range", label_visibility="collapsed"
            )

            adv_from_date, adv_to_date = None, None
            now_utc = datetime.now(timezone.utc)
            adv_to_date = now_utc # 'To' is generally today (or the specified end date)

            if date_range == "Custom range":
                # Use date_input for custom range selection
                adv_from_date_input = st.date_input("From date", key="adv_search_start_date", value=now_utc - timedelta(days=30), max_value=now_utc)
                adv_to_date_input = st.date_input("To date", key="adv_search_end_date", value=now_utc, min_value=adv_from_date_input, max_value=now_utc)
                # Convert selected dates to string format YYYY-MM-DD
                if adv_from_date_input: adv_from_date = adv_from_date_input.strftime('%Y-%m-%d')
                if adv_to_date_input: adv_to_date = adv_to_date_input.strftime('%Y-%m-%d')
            else:
                 # Calculate 'from' date for predefined ranges
                 if date_range == "Last 24 hours":
                      adv_from_date = (now_utc - timedelta(days=1))
                 elif date_range == "Last 7 days":
                      adv_from_date = (now_utc - timedelta(days=7))
                 elif date_range == "Last 30 days":
                      adv_from_date = (now_utc - timedelta(days=30))
                 # Format dates
                 if adv_from_date: adv_from_date = adv_from_date.strftime('%Y-%m-%d')
                 adv_to_date = adv_to_date.strftime('%Y-%m-%d')


        # --- Sort By Option ---
        sort_by = st.selectbox(
            "**Sort results by**",
             ["relevancy", "popularity", "publishedAt"], index=0, # Default relevancy
             key="adv_sort_by",
             help="'Relevancy': Best match to keywords. 'Popularity': Most referenced sources. 'PublishedAt': Newest first."
        )

        # --- Submit Button ---
        submitted = st.form_submit_button("Search Articles")

    # --- Process Advanced Search ---
    if submitted:
        if not search_query or not search_query.strip():
             st.warning("Please enter keywords or phrases to search for.")
             return # Exit if query is empty

        with st.spinner(f"Searching for '{search_query}'..."):
            # --- Build the Final Query String ---
            # Start with the user's base query
            final_adv_query = search_query.strip()

            # Append category constraints if any categories were selected
            if selected_category_keys:
                 category_query_parts = []
                 for cat_key in selected_category_keys:
                     if cat_key in CATEGORIES:
                         # Include main category key and subcategory terms if they exist
                         terms_for_cat = [cat_key.lower()] # Start with main key
                         if 'subcategories' in CATEGORIES[cat_key]:
                             # Add subcategory *values* (the search terms)
                             sub_terms = list(CATEGORIES[cat_key]['subcategories'].values())
                             # Filter out internal identifiers
                             sub_terms = [term for term in sub_terms if term and not term.startswith('_')]
                             terms_for_cat.extend(sub_terms)

                         # Join terms for this category with OR, enclose in parentheses
                         if terms_for_cat:
                             # Remove duplicates just in case
                             terms_for_cat = list(dict.fromkeys(terms_for_cat))
                             # Quote terms with spaces if necessary (optional, NewsAPI might handle it)
                             # quoted_terms = [f'"{t}"' if ' ' in t else t for t in terms_for_cat]
                             # category_query_parts.append(f"({' OR '.join(quoted_terms)})")
                             category_query_parts.append(f"({' OR '.join(terms_for_cat)})")


                 if category_query_parts:
                     # Combine base query with category constraints using AND
                     # Ensure base query is also in parentheses if it contains spaces or logic
                     if " OR " in final_adv_query or " AND " in final_adv_query or " NOT " in final_adv_query:
                          base_query_part = f"({final_adv_query})"
                     else:
                          base_query_part = final_adv_query

                     final_adv_query = f"{base_query_part} AND ({' OR '.join(category_query_parts)})"


            # --- Fetch Results ---
            print(f"Advanced Search Query: {final_adv_query}") # Log the final query
            print(f"Date Range: {adv_from_date} to {adv_to_date}, Sort: {sort_by}")
            articles = fetch_news(
                query=final_adv_query,
                page_size=50, # Fetch a larger number for advanced search display
                from_date=adv_from_date,
                to_date=adv_to_date,
                sort_by=sort_by
            )

            # --- Display Results ---
            if articles:
                st.success(f"Found {len(articles)} articles matching your advanced search.")
                # Highlight based on the original user query terms
                highlight_terms = search_query.split()
                # Reset view tracking for this search results context
                st.session_state.view_tracked = st.session_state.get('view_tracked', set())
                for article in articles:
                    display_article(article, user_id, search_terms=highlight_terms)
            else:
                st.warning("No articles found matching your advanced search criteria. Try refining your keywords, categories, or date range.")

    # elif submitted and not search_query: # Handled at the start of submitted block
    #     st.warning("Please enter keywords or phrases to search for.")


def main_app(user_id):
    """Main application container shown after successful login."""
    st.title("📰 Personalized News Explorer")
    # st.write(f"Logged in as: {st.session_state.user['email']} (UID: {user_id})") # Optional debug info

    # --- Notification Panel (Displayed only if enabled in prefs) ---
    notification_panel_component(user_id)

    # --- Main Content Tabs ---
    tab_titles = ["📰 News Feed", "🔍 Advanced Search"]
    tab1, tab2 = st.tabs(tab_titles)

    with tab1:
        display_regular_news_feed(user_id)

    with tab2:
        display_advanced_search_interface(user_id)

# --- Main Execution Logic ---
def main():
    # --- Initial Checks ---
    # 1. Check if Firebase was initialized successfully earlier
    if not st.session_state.get("firebase_initialized", False):
        st.error("Application cannot start due to Firebase initialization failure. Please check configuration and server logs.")
        st.stop()

    # 2. Perform a lightweight check to ensure Firebase connection is still valid (e.g., list 1 user)
    # This helps catch permission issues or network problems after initialization.
    try:
        auth.list_users(max_results=1)
    except exceptions.FirebaseError as e:
         st.error(f"Firebase connection error after initialization: {str(e)}. Please ensure the service account has permissions and network access.")
         st.stop()
    except Exception as e: # Catch other potential errors like network issues
         st.error(f"Error during Firebase connection check: {str(e)}.")
         st.stop()

    # 3. Check API Key
    if not API_KEY:
         st.error("NewsAPI Key is missing. Please configure NEWSAPI_KEY in your environment or Streamlit secrets.")
         # Allow app to load partially maybe? Or stop? Let's stop for now.
         st.stop()


    # --- Authentication ---
    # The auth_component handles login/signup/logout logic and updates st.session_state.user
    auth_component()

    # --- Conditional Content Display ---
    # Show the main application content only if a user is logged in
    if st.session_state.get('user') and st.session_state.user.get('uid'):
        main_app(st.session_state.user['uid'])
    else:
        # --- Content for Non-Logged-In Users ---
        st.info("Please **Login** or **Sign Up** using the sidebar to access personalized news features and save your preferences.")

        # Optionally display a generic news feed (e.g., top US headlines)
        st.markdown("---")
        st.subheader("🇺🇸 Top US Headlines")
        with st.spinner("Loading trending headlines..."):
            # Fetch a few headlines using the function
            # fetch_top_headlines now returns a list if page_size > 1
            trending_articles = fetch_top_headlines(country='us', page_size=5)

        if trending_articles:
             # Display first few headlines simply for logged-out users
             for article in trending_articles[:3]: # Display max 3
                 if article:
                     with st.container():
                         col1, col2 = st.columns([1, 4])
                         with col1:
                              if article.get('urlToImage'): st.image(article['urlToImage'], use_container_width=True)
                              else: st.image(PLACEHOLDER_IMAGE, use_container_width=True)
                         with col2:
                              st.markdown(f"**[{article.get('title')}]({article.get('url')})**")
                              st.caption(f"{article.get('source',{}).get('name','Unknown Source')} • {format_date(article.get('publishedAt'))}")

                              # --- Corrected Description Handling ---
                              description_raw = article.get('description') # Get value, might be None
                              # Provide a default string if description_raw is None or an empty string
                              description = description_raw if description_raw else 'No description available.'

                              # Truncate and add ellipsis only if the description is longer than the limit
                              max_len = 150
                              if len(description) > max_len:
                                  description_display = description[:max_len] + "..."
                              else:
                                  description_display = description

                              st.write(description_display)
                              # --- End of Correction ---

                     st.markdown("---")

        else:
             st.warning("Could not load trending headlines at this time.")


    # --- Footer ---
    st.markdown("---")
    st.caption("Powered by [NewsAPI.org](https://newsapi.org) | Built with [Streamlit](https://streamlit.io)")
    # Add your name or repo link if desired
    # st.caption("Developed by [Your Name/Link]")


if __name__ == "__main__":
    main()

# --- END OF FILE test.py ---