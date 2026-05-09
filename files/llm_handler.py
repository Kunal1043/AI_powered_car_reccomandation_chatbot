import os
import json
from typing import Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMHandler:
    """Handles LLM interactions for preference extraction."""

    def __init__(self, api_key: Optional[str] = None):

        self.api_key = api_key or os.getenv("HF_API_TOKEN")

        if not self.api_key:
            print("⚠️ No Hugging Face API key found!")
            print("Get free token from: https://huggingface.co/settings/tokens")

        # OpenAI-compatible Hugging Face Router
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=self.api_key,
        )

    def _build_extraction_prompt(self, user_query: str) -> str:
        """Build optimized prompt for preference extraction."""

        return f"""
You are a car recommendation assistant.

Extract car preferences from the user's query.

USER QUERY:
"{user_query}"

TASK:
Extract preferences as valid JSON.

Extract all explicitly mentioned car preferences including brand names.if some fields are not mentioned, leave them out.
for example, if the user does not mention a budget, do not include the "budget_max" field in the JSON.
if there only model name is mentioned without brand,leave out the brand field and only include the model field in the JSON.

Possible fields:
- brand
- model
- year
- budget_max (number in INR)
- fuel_type
- transmission
- min_mileage
- min_seating
- prioritize
Brands in dataset include:['Toyota', 'Kia', 'Maruti Suzuki', 'Honda', 'Volkswagen', 'Renault',
       'Mahindra', 'Tata Motors', 'Skoda', 'Hyundai']

models in dataset include: ['Innova', 'EV6', 'Dzire', 'Amaze', 'City', 'Sonet', 'Glanza',
       'Taigun', 'Lodgy', 'Jazz', 'Civic', 'Kiger', 'Scorpio', 'Punch',
       'Fortuner', 'Kushaq', 'Tiago', 'Superb', 'Camry', 'XUV700',
       'Ertiga', 'Altroz', 'Harrier', 'Carnival', 'Polo', 'Swift',
       'Verna', 'Slavia', 'Nexon', 'Virtus', 'Tiguan', 'Octavia',
       'Seltos', 'Bolero', 'Thar', 'i10', 'Venue', 'Rapid', 'Creta',
       'WR-V', 'XUV300', 'i20', 'Urban Cruiser', 'Vento', 'Baleno',
       'Carens', 'Kwid', 'Duster', 'Triber', 'WagonR']

add model and brand names as given above if any short name or spelling mistake is found in the user query. for example, if the user query contains "tata nexon", extract "Tata" as brand and "Nexon" as model.
for example, tata rather than tata motors, and maruti rather than maruti suzuki.in joson file include tata as tata motors and maruti as maruti suzuki.
use brand and model names as given in the dataset for extraction. if the user query contains "tata nexon", extract "Tata Motors" as brand and "Nexon" as model.

Allowed fuel types:
Petrol, Diesel, CNG, Electric, LPG

Allowed transmission:
Manual, Automatic

Allowed priorities:
comfort, performance, economy, luxury

Return ONLY valid JSON.

Example format:

{{
  "brand": "Tata",
  "model": "Nexon",
  "year": 2023,
  "budget_max": 1500000,
  "fuel_type": "Petrol",
  "transmission": "Automatic",
  "min_mileage": 18,
  "min_seating": 5,
  "prioritize": ["economy"]
}}
extract 
"""

    def extract_preferences(self, user_query: str) -> Dict:
        """Extract structured preferences using LLM."""

        if not self.api_key:
            print("⚠️ Using fallback extraction")
            return self._simple_extraction(user_query)

        try:
            prompt = self._build_extraction_prompt(user_query)

            completion = self.client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct:featherless-ai",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=200,
                temperature=0.3
            )
            generated = completion.choices[0].message.content
            json_str = self._extract_json(generated)
            json_str = json_str.strip()

            json_str = json_str.replace("'", '"')

            preferences = json.loads(json_str)

            return preferences

        except Exception as e:
            print(f"⚠️ Error: {str(e)[:100]}")
            print("Using fallback extraction...")

            return self._simple_extraction(user_query)

    def _extract_json(self, text: str) -> str:
        """Extract JSON object from text."""

        start = text.find("{")
        end = text.rfind("}") + 1

        if start != -1 and end > start:
            return text[start:end]

        return "{}"

    def _simple_extraction(self, query: str) -> Dict:
        """Fallback keyword-based extraction."""

        query_lower = query.lower()

        prefs = {}

        import re

        # Budget
        budget_match = re.search(r'(\d+)\s*lakh', query_lower)

        if budget_match:
            prefs["budget_max"] = int(budget_match.group(1)) * 100000 

        # Fuel type
        fuel_types = ['petrol', 'diesel', 'cng', 'electric', 'lpg']

        for fuel in fuel_types:
            if fuel in query_lower:
                prefs["fuel_type"] = fuel.capitalize()
                break

        # Transmission
        if "automatic" in query_lower:
            prefs["transmission"] = "Automatic"

        elif "manual" in query_lower:
            prefs["transmission"] = "Manual"

        # Mileage
        mileage_match = re.search(r'(\d+)\s*km/l', query_lower)

        if mileage_match:
            prefs["min_mileage"] = int(mileage_match.group(1))

        # Seating
        seating_match = re.search(r'(\d+)\s*seat', query_lower)

        if seating_match:
            prefs["min_seating"] = int(seating_match.group(1))

        # Priorities
        priorities = []

        if any(w in query_lower for w in ['comfort', 'spacious', 'luxury']):
            priorities.append("comfort")

        if any(w in query_lower for w in ['fast', 'performance', 'speed']):
            priorities.append("performance")

        if any(w in query_lower for w in ['economy', 'fuel', 'mileage', 'efficient']):
            priorities.append("economy")

        if priorities:
            prefs["prioritize"] = priorities

        return prefs

    def generate_recommendation_text(self, car_data: Dict, rank: int) -> str:
        """Generate formatted recommendation text."""

        text = f"""
🚗 Recommendation #{rank}

{car_data['Brand']} {car_data['Model']} ({car_data['Year']})

💰 Price: ₹{car_data['Price']:,.0f}

⛽ Fuel: {car_data['Fuel_Type']}
⚙️ Transmission: {car_data['Transmission']}
🛣️ Mileage: {car_data['Mileage']} km/l
🔧 Engine: {car_data['Engine_CC']} cc
🪑 Seats: {int(car_data['Seating_Capacity'])}
🛠️ Service Cost: ₹{car_data['Service_Cost']:,.0f}
"""

        return text.strip()


if __name__ == "__main__":

    handler = LLMHandler()

    test_query = "I want an automatic  tata petrol car under 1500000 with good mileage and 2023 model. I prioritize economy and comfort."  # Example query

    print(f"\nQuery: {test_query}\n")

    prefs = handler.extract_preferences(test_query)

    print("Extracted Preferences:\n")

    print(json.dumps(prefs, indent=2))