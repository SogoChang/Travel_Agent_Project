from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import json

class ScenicSpotSelectionRAG:
    def __init__(self):
        try:
            self.embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            self.rag = FAISS.load_local("scenic_spot_selection_index", self.embeddings_model, allow_dangerous_deserialization=True)
            print("Loaded existing Scenic Spot Selection RAG.")
        except (FileNotFoundError, RuntimeError) as e:
            print("Scenic Spot Selection RAG not found. Initializing new RAG...")
            self.rag = FAISS.from_texts(["Initializing Scenic Spot Selection RAG."], self.embeddings_model)

    def add_guide(self, guide_text: str):
        # Split guide text into individual lines for better retrieval
        guide_lines = guide_text.split("\n")
        guide_lines = [line.strip() for line in guide_lines if line.strip()]
        self.rag.add_texts(guide_lines)
        self.rag.save_local("scenic_spot_selection_index")

    def retrieve_selection_guide(self, query: str) -> str:
        results = self.rag.similarity_search_with_score(query, k=2)
        matched_guides = []

        for doc, score in results:
            print(score)    
            matched_guides.append(doc.page_content)

        if matched_guides:
            return "\n".join(matched_guides)
        return "No suitable guide found."

selection_rag = ScenicSpotSelectionRAG()

# Load and add the Travel Spot Selection Guide
with open("travel_spot_selection_guide.txt", "r", encoding="utf-8") as f:
    guide_text = f.read()

selection_rag.add_guide(guide_text)

# Example Usage
# print("\nRetrieving Selection Guide...")
# guide_info = selection_rag.retrieve_selection_guide("What are the best spots for adventure lovers?")
# print(guide_info)
