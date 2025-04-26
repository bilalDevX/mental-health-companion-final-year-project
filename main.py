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


from sqlalchemy import Column, Integer, String, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from databases import Database


# Database Config
DATABASE_URL = "sqlite:///./mental_health.db"
database = Database(DATABASE_URL)
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Define ChatHistory Table
class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    primary_emotion = Column(String, nullable=False)
    crisis_detected = Column(Boolean, default=False)

# Create Table
Base.metadata.create_all(bind=engine)

# Save Chat Function
async def save_chat(text, emotion, crisis):
    async with database.transaction():
        query = ChatHistory.__table__.insert().values(
            text=text, primary_emotion=emotion, crisis_detected=crisis
        )
        await database.execute(query)






class TextInput(BaseModel):
    text: str

@app.post("/analyze_emotions")
async def analyze_emotions(input_text: TextInput):

    results = emotion_pipeline(input_text.text)
    emotions = {res["label"]: res["score"] for res in results[0]}  
    primary_emotion = max(emotions, key=emotions.get)  
    coping_strategy = COPING_STRATEGIES.get(primary_emotion, "Stay mindful and take care of yourself.")    
    crisis_detected = any(keyword in input_text.text.lower() for keyword in CRISIS_KEYWORDS)
    crisis_help = CRISIS_RESPONSE if crisis_detected else None
    chat_model = init_chat_model("llama3-8b-8192", model_provider="groq")

    ai_prompt = f"""
     User Input: {input_text.text}
Detected Emotion: {primary_emotion}
Suggested Coping Strategy: {coping_strategy}
Crisis Detected: {'Yes' if crisis_detected else 'No'} 

Instructions for the assistant:

reponse like a therapist:

Act as a professional mental health therapist.
- Respond in a short (1–300 characters), empathetic, theropiest-like tone.
- Use concise sentences; avoid long paragraphs.
- If a crisis is detected, respond calmly and supportively first (e.g., suggest calling a crisis line or confiding in someone trusted).
- Maintain a low response temperature (around 0.1–0.4) for thoughtful answers.
- After replying, casually mention the detected emotion and a coping strategy (not in structured JSON). Repeat the detected emotion after your response for comparison.

Analyze the user input, then:
- Provide a brief empathetic response following the above guidelines.
- State the emotion you detect in the user input.
- Suggest a coping strategy for that emotion.
     """

    response = chat_model.invoke(ai_prompt)

    await save_chat(input_text.text, primary_emotion, crisis_detected)

    return {
        "text": input_text.text,
        "emotions": emotions,
        "primary_emotion": primary_emotion,
        "coping_strategy": coping_strategy,
        "crisis_detected": crisis_detected,
        "crisis_help": crisis_help,
        "ai_response": response.content,
    }


@app.get("/")
async def read_root():
    try:
        # Initialize the Groq model
       model = init_chat_model("llama3-8b-8192", model_provider="groq")
       
       input_query = "Hello, world!"
       
       result = model.invoke(input_query)
   
       return {
            # "message": "Welcome to the FastAPI!",
            # "api_key": "Loaded Successfully",
            "response": result.content,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from sqlalchemy.orm import Session
from fastapi import Depends

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/chat_history")
async def get_chat_history(db: Session = Depends(get_db)):
    messages = db.query(ChatHistory).all()
    return messages
