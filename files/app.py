"""
AI-Powered Car Recommendation Chatbot
A production-level Streamlit application with conversation memory,
intelligent recommendations, and professional UI.

Installation:
    pip install streamlit pandas requests python-dotenv

Run:
    streamlit run app.py

Deployment on Streamlit Cloud:
    1. Push code to GitHub
    2. Go to https://share.streamlit.io
    3. Connect your repo
    4. Add HF_API_TOKEN secret in settings
    5. Deploy!
"""

import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from pathlib import Path

# Import custom modules
from clean_data import DataCleaner, prepare_dataset
from llm_handler import LLMHandler
from recommender import CarRecommender


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🚗 CarBot - AI Car Recommendation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --danger-color: #d62728;
    }
    
    /* Recommendation cards */
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .recommendation-title {
        font-size: 1.3em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .score-badge {
        background: rgba(255,255,255,0.2);
        padding: 5px 10px;
        border-radius: 20px;
        display: inline-block;
        margin: 5px 5px 5px 0;
        font-size: 0.9em;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .chat-user {
        background-color: #e3f2fd;
        border-left: 4px solid #1f77b4;
    }
    
    .chat-assistant {
        background-color: #f5f5f5;
        border-left: 4px solid #2ca02c;
    }
    
    /* Metrics */
    .metric-box {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE & DATA INITIALIZATION
# ============================================================================

@st.cache_resource
def load_dataset():
    """Load and cache the car dataset."""
    data_path = r'D:\\car_chatbot_2\\data\\car_dataset_india.csv'
    df = pd.read_csv(data_path)
   
    return df


@st.cache_resource
def initialize_llm():
    """Initialize LLM handler."""
    api_key = os.getenv('HF_API_TOKEN')
    return LLMHandler(api_key=api_key)

def initialize_session():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'preferences' not in st.session_state:
        st.session_state.preferences = {}
    
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    
    if 'llm_handler' not in st.session_state:
        st.session_state.llm_handler = initialize_llm()
    
    if 'dataset' not in st.session_state:
        st.session_state.dataset = load_dataset()
    
    if 'recommender' not in st.session_state and st.session_state.dataset is not None:
        st.session_state.recommender = CarRecommender(st.session_state.dataset)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def add_to_chat_history(role: str, content: str, preferences: dict = None, recommendations: list = None):
    """Add message to chat history."""
    message = {
        'role': role,
        'content': content,
        'timestamp': datetime.now().isoformat(),
        'preferences': preferences,
        'recommendations': recommendations
    }
    st.session_state.chat_history.append(message)


def display_recommendation_card(rec: dict, index: int):
    """Display a single recommendation as a styled card."""
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"### 🏆 #{index+1}: {rec['Brand']} {rec['Model']}")
            
            # Key details
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Price", f"₹{rec['Price']/100000:.1f}L")
            with col_b:
                st.metric("Mileage", f"{rec['Mileage']} km/l")
            with col_c:
                st.metric("Engine", f"{rec['Engine_CC']} cc")
            with col_d:
                st.metric("Year", rec['Year'])
            
            # Features
            feature_text = f"""
            **Fuel:** {rec['Fuel_Type']} | **Transmission:** {rec['Transmission']} | **Seats:** {rec['Seating_Capacity']}
            
            **Service Cost:** ₹{rec['Service_Cost']:,.0f} per service
            """
            st.markdown(feature_text)
        
        with col2:
            st.metric("Overall Score", f"{rec['Overall_Score']}/100", delta=None)
            st.metric("Match Score", f"{rec['Match_Score']}/100", delta=None)
        
        with col3:
            st.metric("Value Score", f"{rec['Value_Score']}/100", delta=None)
            if rec['Matched_Preferences']:
                st.success("✓ " + "\n✓ ".join(rec['Matched_Preferences'][:3]))
        
        st.divider()


def display_chat_history():
    """Display conversation history."""
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.markdown(f"**👤 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 CarBot:** {msg['content']}")
            
            # Display recommendations if available
            if msg.get('recommendations'):
                st.info("📋 Recommendations found!")
        
        st.markdown("---")


def process_user_query(user_query: str):
    """
    Process user query and generate recommendations.
    
    Steps:
    1. Extract preferences using LLM
    2. Filter recommendations
    3. Score and rank cars
    4. Format response
    """
    
    # Show loading state
    with st.spinner("🤔 Analyzing your query..."):
        # Step 1: Extract preferences
        preferences = st.session_state.llm_handler.extract_preferences(user_query)
        st.session_state.preferences = preferences
        
        # Log extracted preferences
        st.success(f"✓ Extracted preferences: {json.dumps(preferences, indent=2)}")
    
    with st.spinner("🔍 Searching for matching cars..."):
        # Step 2: Get recommendations
        recommendations = st.session_state.recommender.recommend(
            preferences, 
            num_recommendations=5
        )
        st.session_state.recommendations = recommendations
    
    # Format response
    if recommendations:
        response_text = f"Found {len(recommendations)} great options for you! 🎉"
        add_to_chat_history('assistant', response_text, preferences, recommendations)
        
        # Display recommendations
        st.success(response_text)
        st.markdown("### 🚗 Recommended Cars")
        
        for idx, rec in enumerate(recommendations):
            display_recommendation_card(rec, idx)
    else:
        response_text = "Sorry, I couldn't find cars matching your criteria. Try relaxing some requirements!"
        add_to_chat_history('assistant', response_text)
        st.warning(response_text)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main Streamlit app."""
    
    # Initialize session
    initialize_session()
    
    # Header
    st.title("🚗 CarBot - AI Car Recommendation Engine")
    st.markdown("Find your perfect car using AI-powered recommendations")
    
    # Check if dataset loaded
    if st.session_state.dataset is None or len(st.session_state.dataset) == 0:
        st.error("❌ Dataset not available. Please ensure cleaned_cars.csv exists or upload a file.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Dataset Info")
        st.metric("Total Cars", len(st.session_state.dataset))
        st.metric("Brands", st.session_state.dataset['Brand'].nunique())
        
        # Price range
        min_price = st.session_state.dataset['Price'].min()
        max_price = st.session_state.dataset['Price'].max()
        st.metric("Price Range", f"₹{min_price/100000:.1f}L - ₹{max_price/100000:.1f}L")
        
        # Fuel types
        st.markdown("### ⛽ Available Fuel Types")
        fuel_counts = st.session_state.dataset['Fuel_Type'].value_counts()
        for fuel, count in fuel_counts.items():
            st.write(f"• {fuel}: {count} cars")
        
        # Clear chat
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.preferences = {}
            st.session_state.recommendations = None
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 💬 Tell me what you're looking for")
        
        # Example queries
        with st.expander("📝 Example queries (click to expand)"):
            examples = [
                "I want an automatic petrol car under 15 lakh with good mileage",
                "Show me affordable family cars with at least 5 seats",
                "Looking for a high-performance car with a diesel engine",
                "Budget is 20 lakh, need something fuel-efficient and spacious",
                "I prefer electric or hybrid vehicles under 25 lakh"
            ]
            for example in examples:
                if st.button(f"📌 {example}", use_container_width=True):
                    st.session_state.user_query = example
    
    # Chat input
    user_input = st.text_area(
        "Your Query:",
        placeholder="Describe what car you're looking for...",
        height=80,
        key="user_input"
    )
    
    col_search, col_clear = st.columns([4, 1])
    with col_search:
        search_button = st.button("🔍 Find Cars", use_container_width=True, type="primary")
    with col_clear:
        if st.button("❌", use_container_width=True):
            st.session_state.user_input = ""
    
    # Process query
    if search_button and user_input:
        # Add to chat
        add_to_chat_history('user', user_input)
        
        # Process
        process_user_query(user_input)
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📋 Conversation History")
        display_chat_history()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### ℹ️ About CarBot
    This AI-powered chatbot helps you find the perfect car based on your preferences.
    
    **How it works:**
    1. Describe what you're looking for in natural language
    2. AI extracts your preferences (budget, fuel type, features, etc.)
    3. We filter and score cars from our database
    4. Top matches are displayed with detailed information
    """)


if __name__ == "__main__":
    main()
