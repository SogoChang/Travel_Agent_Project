from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool, StructuredTool
import google.generativeai as genai
import os
import requests
import json
import time

class ScenicSpotRAG:
    def __init__(self):
        # Load or create a FAISS index
        try:
            self.embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            self.rag = FAISS.load_local("scenic_spot_index", self.embeddings_model, allow_dangerous_deserialization=True)
            print("Loaded existing FAISS RAG.")
        except (FileNotFoundError, RuntimeError) as e:
            print("FAISS index not found. Initializing new RAG...")
            initial_texts = ["Initializing Scenic Spot RAG."]
            self.rag = FAISS.from_texts(initial_texts, self.embeddings_model)

        # Initialize Gemini LLM
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            print("Gemini model initialized.")
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
            self.gemini_model = None

    @tool("tripadvisor_location_search", description="Search for location IDs on TripAdvisor based on a query.")
    def tripadvisor_location_search(query: str, category: str = "attractions", language: str = "en") -> list:
        api_key = os.getenv("TRIPADVISOR_API_KEY")
        url = f"https://api.content.tripadvisor.com/api/v1/location/search?language={language}&key={api_key}&searchQuery={query}&category={category}"
        response = requests.get(url, headers={"accept": "application/json"})
        if response.status_code == 200:
            return response.json().get("data", [])
        return []

    @tool("tripadvisor_location_detail", description="Retrieve detailed information of a location using TripAdvisor API and Location ID.")
    def tripadvisor_location_detail(location_id: str, language: str = "en", currency: str = "USD") -> dict:
        api_key = os.getenv("TRIPADVISOR_API_KEY")
        url = f"https://api.content.tripadvisor.com/api/v1/location/{location_id}/details?language={language}&currency={currency}&key={api_key}"
        response = requests.get(url, headers={"accept": "application/json"})
        return response.json() if response.status_code == 200 else {}

    @tool("google_place_id_search", description="Search for Google Place ID using Google Maps Places API.")
    def google_place_id_search(search_text: str) -> str:
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        url = "https://places.googleapis.com/v1/places:searchText"
        headers = {"Content-Type": "application/json", "X-Goog-Api-Key": api_key, "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location"}
        data = {"textQuery": search_text, "languageCode": "en"}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        results = response.json().get("places", [])
        return results[0].get("id", "No Place ID found") if results else "No Place ID found"

    @tool("google_place_detail_search", description="Retrieve detailed information of a place using Google Maps Places API and Place ID.")
    def google_place_detail_search(place_id: str) -> dict:
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        url = f"https://places.googleapis.com/v1/places/{place_id}"
        #headers = {"Content-Type": "application/json", "X-Goog-Api-Key": api_key, "X-Goog-FieldMask": "places.id,places.types,places.formattedAddress,places.location,places.viewport,places.rating,places.googleMapsUri,places.places.websiteUri,places.regularOpeningHours,places.displayName,places.primaryTypeDisplayName,places.currentOpeningHours,places.editorialSummary,places.review,"}
        headers = {"Content-Type": "application/json", "X-Goog-Api-Key": api_key, "X-Goog-FieldMask": "*"}
        response = requests.get(url, headers=headers)
        return response.json() if response.status_code == 200 else {}

    def generate_description(self, combined_info: dict) -> str:
        if not self.gemini_model:
            return "Gemini model not initialized."

        prompt = f"Generate a detailed, vivid description for the scenic spot {combined_info['name']} using the following information.\n\nTripAdvisor Data: {combined_info['tripadvisor']}\nGoogle Data: {combined_info['google']} Also try to add what's the best time to visit into the description."

        generation_config = genai.types.GenerationConfig(
                # candidate_count=1, # 通常預設為 1
                # stop_sequences=['\n\n\n'], # 可選的停止序列
                max_output_tokens=250, # 調整最大輸出 token 數量
                temperature=0.7 # 調整生成文本的創意度，0.0-1.0
            )
        response = self.gemini_model.generate_content(prompt, generation_config=generation_config)

        if response and response.parts:
            return response.text.strip()
        return "Failed to generate description."

    def retrieve_scenic_spot(self, query: str) -> str:
        print(f"Attempting to retrieve scenic spot: {query}")
        results = self.rag.similarity_search_with_score(query, k=10)
        all_details = []

        for doc, score in results:
            if score < 0.85 and os.path.exists(doc.page_content.strip()):
                print(f"Loaded from RAG {score}")
                with open(doc.page_content.strip(), 'r') as f:
                    data = json.load(f)
                    all_details.append({"name": data.get("name"), "description": data.get("description")})

        if all_details:
            return json.dumps(all_details, ensure_ascii=False)

        print("Loading from API...")
        locations = self.tripadvisor_location_search.run({"query": query})

        for loc in locations:
            google_id = self.google_place_id_search.run({"search_text": loc['name']})
            google_detail = self.google_place_detail_search.run({"place_id": google_id})
            trip_detail = self.tripadvisor_location_detail.run({"location_id": loc['location_id']})

            combined_info = {
                "name": loc['name'],
                "tripadvisor": trip_detail,
                "google": google_detail
            }

            combined_info["description"] = self.generate_description(combined_info)
            json_path = f"scenic_spots/{loc['name'].replace(' ', '_')}.json"
            with open(json_path, 'w') as f:
                json.dump(combined_info, f, indent=4)

            self.rag.add_texts([json_path])  # Only save JSON path to RAG
            self.rag.save_local("scenic_spot_index")
            all_details.append({"name": combined_info['name'], "description": combined_info['description']})
            time.sleep(2)

        return json.dumps(all_details, ensure_ascii=False)
    
    def filter_spots_by_preference(self, spot_info: list, preference: str) -> list:
        from scenic_spot_selection_rag import selection_rag
        guide_text = selection_rag.retrieve_selection_guide(preference)

        if guide_text == "No suitable guide found.":
            return []

        filtered_spots = []

        for spot in spot_info:
            prompt = f"Based on the guide: {guide_text}\nShould this scenic spot be recommended for {preference} travelers?\nSpot Description: {spot['description']}\nRespond with 0 for NO and 1 for YES. Only response 0 or 1, nothing else."
            generation_config = genai.types.GenerationConfig(
                # candidate_count=1, # 通常預設為 1
                # stop_sequences=['\n\n\n'], # 可選的停止序列
                max_output_tokens=250, # 調整最大輸出 token 數量
                temperature=0.7 # 調整生成文本的創意度，0.0-1.0
            )
            response = self.gemini_model.generate_content(prompt, generation_config=generation_config)
            answer = response.text.strip() if response and response.text else "0"
            #print(answer)
            if answer == "1":
                print(f"add {spot['name']}")
                filtered_spots.append(spot['name'])
            else:
                print(f"do not add {spot['name']}")
            time.sleep(1)
        return filtered_spots

# Initialize the RAG
scenic_spot_rag = ScenicSpotRAG()

# Example Usage
print("\nRetrieving Scenic Spot...")
spot_info = scenic_spot_rag.retrieve_scenic_spot("Attractions in Tainan")
#print(spot_info)
filtered_spots = scenic_spot_rag.filter_spots_by_preference(json.loads(spot_info), "Cultural Explorers")
print(filtered_spots)
