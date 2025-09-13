import os
import streamlit as st
from dotenv import load_dotenv
from serpapi import GoogleSearch
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load API keys
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Initialize Ollama LLM
llm = Ollama(model="llama3.2", temperature=0.7)

# Prompt Template with flights
prompt_template = PromptTemplate(
    input_variables=["departure", "destination", "start_date", "end_date", "budget", "interests", "places", "weather", "hotels", "flights"],
    template="""
You are an AI Travel Planner.  
Your task is to create a **realistic and detailed day-wise itinerary**.

### Trip Details:
- Departure City: {departure}  
- Destination City: {destination}  
- Travel Dates: {start_date} to {end_date}  
- Total Budget: Rs {budget}  
- Traveler Interests: {interests}  

### Available Information:
- Recommended Attractions: {places}  
- Weather Forecast: {weather}  
- Hotel Options: {hotels}  
- Flight Options: {flights}  

---

### Instructions:
1. Create a **day-wise itinerary** in table format.  
2. Each day should include Morning, Afternoon, and Evening activities.  
3. Suggest hotels, food, and give weather-based tips.  
4. Show estimated costs in INR per day.  
5. At the end, provide a **summary table** of the trip with total cost, recommended hotel, and best flight option.  

---

### Output Format:

#### Day-Wise Itinerary
| Day | Morning Activities | Afternoon Activities | Evening Activities | Hotel Suggestion | Food Recommendations | Tips | Estimated Cost (INR) |
|-----|--------------------|----------------------|-------------------|------------------|----------------------|------|----------------------|

(Fill this table for each day of the trip)

---

#### Trip Summary
| Item                | Recommendation |
|---------------------|----------------|
| Total Estimated Cost | ... |
| Best Flight Option   | ... |
| Recommended Hotel    | ... |
| Key Highlights       | ... |

    """,
)

travel_chain = LLMChain(llm=llm, prompt=prompt_template)

# Streamlit UI
st.set_page_config(page_title="AI Travel Planner", page_icon="✈️")
st.title("🧳 AI-Powered Travel Planner (Ollama + SerpAPI)")

departure = st.text_input("Where are you flying from? (City or Airport Code)")
destination = st.text_input("Where do you want to go?")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")
budget = st.text_input("Budget (Rs)")
interests = st.text_area("Interests (e.g., adventure, food, culture)")

if st.button("Generate Itinerary"):
    with st.spinner("Fetching attractions + hotels + flights + weather + generating plan..."):
        try:
            # --- 1. Attractions from SerpAPI ---
            attractions = []
            try:
                params = {
                    "engine": "google",
                    "q": f"Top tourist attractions in {destination}",
                    "api_key": SERPAPI_KEY,
                }
                search = GoogleSearch(params)
                results = search.get_dict()
                attractions = [res.get("title") for res in results.get("organic_results", [])[:5]]
            except Exception:
                attractions = ["No attractions found"]

            # --- 2. Hotels from SerpAPI ---
            hotels = []
            try:
                params = {
                    "engine": "google_hotels",
                    "q": f"Hotels in {destination}",
                    "check_in_date": str(start_date),
                    "check_out_date": str(end_date),
                    "api_key": SERPAPI_KEY,
                }
                search = GoogleSearch(params)
                results = search.get_dict()
                hotels = [f"{hotel.get('name')} - {hotel.get('rate_per_night', {}).get('lowest', 'N/A')} INR"
                          for hotel in results.get("properties", [])[:5]]
            except Exception:
                hotels = ["No hotels found"]

            # --- 3. Flights from SerpAPI ---
            flights = []
            try:
                params = {
                    "engine": "google_flights",
                    "departure_id": departure,  # user input (IATA code works best, e.g., DEL, BOM, JFK)
                    "arrival_id": destination,
                    "outbound_date": str(start_date),
                    "return_date": str(end_date),
                    "currency": "INR",  # ✅ Prices in INR
                    "api_key": SERPAPI_KEY,
                }
                search = GoogleSearch(params)
                results = search.get_dict()
                flights = [
                    f"{flight.get('airline')} - {flight.get('price')} INR"
                    for flight in results.get("best_flights", [])[:5]
                ]
            except Exception:
                flights = ["No flights found"]

            # --- 4. Weather from SerpAPI ---
            weather = "Not available"
            try:
                params = {
                    "engine": "google",
                    "q": f"Weather in {destination}",
                    "api_key": SERPAPI_KEY,
                }
                search = GoogleSearch(params)
                results = search.get_dict()
                weather = results.get("answer_box", {}).get("temperature") or "Weather info not found"
            except Exception:
                weather = "Weather API error"

            # --- 5. Generate itinerary with Ollama ---
            itinerary = travel_chain.run(
                departure=departure,
                destination=destination,
                start_date=start_date,
                end_date=end_date,
                budget=budget,
                interests=interests,
                places=", ".join(attractions),
                weather=weather,
                hotels=", ".join(hotels),
                flights=", ".join(flights),
            )

            # --- 6. Display ---
            st.success("✅ Itinerary generated!")
            st.subheader("📍 Suggested Attractions")
            for place in attractions:
                st.write(f"- {place}")
            st.subheader("🏨 Hotel Options")
            for hotel in hotels:
                st.write(f"- {hotel}")
            st.subheader("✈️ Flight Options")
            for flight in flights:
                st.write(f"- {flight}")
            st.subheader("🌦 Weather Forecast")
            st.write(weather)
            st.subheader("📝 Itinerary")
            st.markdown(itinerary)

        except Exception as e:
            st.error(f"Error: {e}")
