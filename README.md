# AI-Powered Car Recommendation Chatbot

An intelligent conversational car recommendation system that understands natural language queries and suggests the most suitable cars based on user preferences such as budget, brand, fuel type, transmission, mileage, and seating capacity.

## Demo Query

> "I want a Honda Civic automatic petrol car under 15 lakh with good mileage."

## Features

- Natural language understanding using Large Language Models (LLMs)
- Extracts user preferences automatically from text queries
- Filters cars by:
  - Brand and model
  - Budget
  - Fuel type
  - Transmission
  - Mileage
  - Seating capacity
- Intelligent scoring and ranking of recommendations
- Fallback rule-based extraction if LLM response fails
- Interactive Streamlit web interface

## Tech Stack

- Python
- Streamlit
- Pandas
- OpenAI Python SDK
- Hugging Face Router API
- Llama 3.1 / Qwen models
- python-dotenv

## Project Structure

```text
CAR_CHATBOT_2/
├── data/
│   └── car_dataset_india.csv
│
├── files/
│   ├── app.py
│   ├── recommender.py
│   ├── llm_handler.py
│   └── clean_data.py
│
├── README.md
├── requirements.txt
├── .gitignore
└── .env.example