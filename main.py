from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set. Please define it in the .env file.")

app = FastAPI()

emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=3)




COPING_STRATEGIES = {
    "anger": "Try deep breathing, meditation, or physical exercise to release stress.",
    "disgust": "Engage in positive self-talk and avoid toxic environments.",
    "fear": "Practice mindfulness, grounding techniques, or talk to a trusted person.",
    "joy": "Enjoy the moment, express gratitude, and share your happiness with others!",
    "neutral": "Maintain a balanced lifestyle and engage in self-care activities.",
    "sadness": "Reach out to friends, journal your feelings, or try relaxing activities.",
    "surprise": "Embrace change, reflect on new opportunities, and stay adaptable."
}

CRISIS_KEYWORDS = {"suicide", "self-harm", "kill myself", "no way out", "hopeless", "die", "end it all", "worthless"}

CRISIS_RESPONSE = {
    "hotline": "If you're in distress, please call a mental health helpline. In the US: 988 Suicide & Crisis Lifeline.",
    "advice": "You're not alone. Talk to someone you trust or seek professional help."
}

class TextInput(BaseModel):
    text: str

@app.post("/analyze_emotions")
def analyze_emotions(input_text: TextInput):
    results = emotion_pipeline(input_text.text)
    
    # Get top 3 detected emotions
    emotions = {res["label"]: res["score"] for res in results[0]}  
    primary_emotion = max(emotions, key=emotions.get)  

    coping_strategy = COPING_STRATEGIES.get(primary_emotion, "Stay mindful and take care of yourself.")

    
    # Crisis Detection
    crisis_detected = any(keyword in input_text.text.lower() for keyword in CRISIS_KEYWORDS)

    crisis_help = CRISIS_RESPONSE if crisis_detected else None



    return {
        "text": input_text.text,
        "emotions": emotions,
        "primary_emotion": primary_emotion,
        "coping_strategy": coping_strategy,
        "crisis_detected": crisis_detected,
        "crisis_help": crisis_help
    }


@app.get("/")
def read_root():
    try:
        # Initialize the Groq model
       model = init_chat_model("llama3-8b-8192", model_provider="groq")
       input_query = "Hello, world!"
       
       result = model.invoke(input_query)
       emotion = analyze_emotions(TextInput(text= input_query))
       return {
            # "message": "Welcome to the FastAPI!",
            # "api_key": "Loaded Successfully",
            "response": result.content,
            "emotion": emotion
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


