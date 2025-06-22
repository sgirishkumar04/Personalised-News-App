# Newsify - A Personalized News Aggregator 📰

[![Streamlit](https://img.shields.io/badge/Streamlit-1.44-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=for-the-badge&logo=firebase)](https://firebase.google.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)

Tired of generic news feeds? **Newsify** is an intelligent news aggregator built with Streamlit that learns from your behavior to deliver a truly personalized news experience. It features robust user authentication, real-time recommendations, and a clean, modern interface.

---

## ✨ Key Features

-   **🧠 Personalized "For You" Feed:** The core of the app. It uses **TF-IDF and Cosine Similarity** to analyze your reading history (likes, views, dislikes) and recommend articles that match your interests.
-   **🔐 User Authentication & Persistence:** Secure user signup and login handled by **Firebase Authentication**. All user preferences, liked articles, and saved-for-later content are persisted in **Cloud Firestore**.
-   **👍 Interactive News Engagement:** Users can **Like**, **Dislike**, and **Save** articles. These interactions directly fuel the recommendation engine.
-   **⚙️ Customizable Preferences:** Users can select their favorite news categories to customize the content used for their "For You" feed and receive relevant breaking news alerts.
-   **🔔 Real-time Breaking News Alerts:** An in-app notification panel alerts logged-in users to the latest top headlines from their preferred categories.
-   **📂 Rich Category Browsing:** Explore news across 7 major categories and dozens of specific sub-topics.
-   **🔍 Advanced Search:** A dedicated search interface with filters for keywords, date ranges, and sorting options.
-   **📧 Email Notifications:** Optional integration to send welcome emails on signup and security alerts on login.

---

## 🤖 How the Personalization Engine Works

The "For You" feed is powered by a classic content-based filtering approach:

1.  **Interaction Tracking:** The app logs every user interaction (like, dislike, view) with articles in Firestore.
2.  **User Profile Vector:** A weighted vector representing the user's interests is created. 'Likes' are heavily weighted, 'views' have a moderate weight, and 'dislikes' have a negative weight.
3.  **Candidate Articles:** A broad set of recent and popular articles are fetched from NewsAPI based on the user's preferred categories.
4.  **TF-IDF Vectorization:** The text (title + description) of both the user's interaction history and the candidate articles is converted into numerical vectors using `TfidfVectorizer`.
5.  **Similarity Scoring:** **Cosine Similarity** is calculated between the user's profile vector and each candidate article vector.
6.  **Ranking & Recency Boost:** Articles are ranked by their similarity score. A recency boost is added to prioritize newer content, ensuring the feed stays fresh.
7.  **Display:** The top-ranked articles are presented to the user in their "For You" feed.

---

## 🛠️ Technology Stack

-   **Framework:** Streamlit
-   **Language:** Python
-   **Database & Auth:** Google Firebase (Authentication, Cloud Firestore)
-   **News Source:** [NewsAPI.org](https://newsapi.org)
-   **Machine Learning / Data:** Scikit-learn, NumPy, Pandas
-   **Environment Management:** python-dotenv
-   **Email:** smtplib, ssl

---

## 🚀 How to Run Locally

### Prerequisites
-   Python 3.11+
-   A [NewsAPI](https://newsapi.org) Key
-   A [Google Firebase](https://firebase.google.com/) project with **Authentication** (Email/Password provider enabled) and **Cloud Firestore** (in Native mode) enabled.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/YOUR_USERNAME/Personalised-News-App.git
    cd Personalised-News-App
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

### Configuration

1.  **Set up Firebase Credentials:**
    -   In your Firebase project console, go to **Project settings** -> **Service accounts**.
    -   Click **"Generate new private key"** to download a JSON credentials file.
2.  **Create the `.env` file:**
    Create a file named `.env` in the root of the project directory. Open the downloaded JSON file and copy the values into the `.env` file like this:
    ```env
    # NewsAPI Key
    NEWSAPI_KEY="YOUR_NEWSAPI_KEY_HERE"

    # Firebase Credentials (copy from your downloaded JSON file)
    FIREBASE_TYPE="service_account"
    FIREBASE_PROJECT_ID="your-project-id"
    FIREBASE_PRIVATE_KEY_ID="your-private-key-id"
    FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nYourPrivateKeyLine1\nYourPrivateKeyLine2\n-----END PRIVATE KEY-----\n"
    FIREBASE_CLIENT_EMAIL="your-client-email@your-project-id.iam.gserviceaccount.com"
    FIREBASE_CLIENT_ID="your-client-id"
    FIREBASE_AUTH_URI="https://accounts.google.com/o/oauth2/auth"
    FIREBASE_TOKEN_URI="https://oauth2.googleapis.com/token"
    FIREBASE_AUTH_PROVIDER_CERT_URL="https://www.googleapis.com/oauth2/v1/certs"
    FIREBASE_CLIENT_CERT_URL="your-client-cert-url"

    # Optional: Email Configuration for Notifications
    # SMTP_SERVER="smtp.example.com"
    # SMTP_PORT="587"
    # SMTP_USERNAME="your-email@example.com"
    # SMTP_PASSWORD="your-email-password"
    # EMAIL_FROM="your-email@example.com"
    ```
    **Important:** For `FIREBASE_PRIVATE_KEY`, ensure you copy the entire key including the `-----BEGIN...` and `-----END...` lines, and preserve the `\n` characters.

### Run the Application

Execute the following command in your terminal:
```sh
streamlit run news_app.py
```

---


## 💡 Challenges & Learnings

Building and deploying this project involved overcoming several real-world challenges, which were fantastic learning opportunities:

-   **Firebase Integration in Streamlit:** Successfully integrating the **Firebase Admin SDK** required careful handling of initialization logic to prevent re-initialization on every Streamlit script rerun, using `st.session_state` to manage the app's connection status.

-   **Real-time Personalization:** Balancing the computational cost of the recommendation engine with a smooth user experience. This involved using `@st.cache_data` for API calls and optimizing the **TF-IDF vectorizer** settings.

-   **State Management:** Managing a complex application state (user login, current category, notifications) within Streamlit's stateless execution model was a key challenge, solved by extensive use of `st.session_state`.
