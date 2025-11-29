from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.services.question_generator import QuestionGenerator

app = FastAPI(
    title="AI LeetCode Platform API",
    description="Backend for the AI-powered adaptive learning platform",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Vercel deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


question_generator = QuestionGenerator()

class GenerateRequest(BaseModel):
    topic: str
    difficulty: str
    user_context: str = None

@app.get("/")
async def root():
    return {"message": "Welcome to the AI LeetCode Platform API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/generate_question")
async def generate_question(request: GenerateRequest):
    try:
        question = await question_generator.generate_question(
            request.topic, 
            request.difficulty, 
            request.user_context
        )
        return question
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
