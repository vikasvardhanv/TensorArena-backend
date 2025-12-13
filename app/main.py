from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.services.question_generator import QuestionGenerator
from app.services.code_executor import CodeExecutor
from app.services.role_question_generator import RoleBasedQuestionGenerator
from app.services.system_design_service import SystemDesignService
from app.services.mentor_service import MentorService
from app.services.deepgram_service import DeepgramService
from app.services.mock_interview_service import MockInterviewService

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
code_executor = CodeExecutor(timeout=5)
role_question_generator = RoleBasedQuestionGenerator()
mentor_service = MentorService()
system_design_service = SystemDesignService()

class GenerateRequest(BaseModel):
    topic: str
    difficulty: str
    user_context: str = None

class ExecuteCodeRequest(BaseModel):
    code: str

class GenerateRoleQuestionsRequest(BaseModel):
    role: str
    count: int = 3

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

@app.post("/execute_code")
async def execute_code(request: ExecuteCodeRequest):
    try:
        result = code_executor.execute_python(request.code)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_role_questions")
async def generate_role_questions(request: GenerateRoleQuestionsRequest):
    """
    Generate role-based scenario questions for specific AI/ML roles.

    Supports roles:
    - Machine Learning Engineer
    - Data Scientist
    - Computer Vision Engineer
    - NLP Engineer
    - LLM Specialist
    - AI Product Manager

    Returns an array of diverse question types (multiple-choice, output-selection, fill-in-blank).
    """
    try:
        questions = await role_question_generator.generate_role_questions(
            request.role,
            request.count
        )
        return questions
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_system_design_question")
async def generate_system_design_question(request: GenerateRequest):
    try:
        question = await question_generator.generate_system_design_question(
            request.topic, 
            request.difficulty
        )
        return question
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_production_question")
async def generate_production_question(request: GenerateRequest):
    try:
        question = await question_generator.generate_production_question(
            request.topic, 
            request.difficulty
        )
        return question
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_paper_question")
async def generate_paper_question(request: GenerateRequest):
    try:
        question = await question_generator.generate_paper_question(
            request.topic, 
            request.difficulty
        )
        return question
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_interview_question")
async def generate_interview_question(request: GenerateRequest):
    try:
        question = await question_generator.generate_interview_question(
            request.topic, 
            request.difficulty
        )
        return question
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class GradeRequest(BaseModel):
    code: str
    question_title: str
    question_description: str
    language: str = "python"

@app.post("/grade_submission")
async def grade_submission(request: GradeRequest):
    try:
        grading = await question_generator.grade_submission(
            request.code,
            request.question_title,
            request.question_description
        )
        return grading
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class MentorChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/mentor/chat")
async def mentor_chat(request: MentorChatRequest):
    try:
        response = await mentor_service.chat(request.message, request.history)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SystemDesignChatRequest(BaseModel):
    message: str
    history: list = []
    topic: str = "General System Design"

@app.post("/system-design/chat")
async def system_design_chat(request: SystemDesignChatRequest):
    try:
        response = await system_design_service.chat(request.message, request.history, request.topic)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

deepgram_service = DeepgramService()
mock_interview_service = MockInterviewService()

@app.get("/deepgram/token")
async def get_deepgram_token():
    return await deepgram_service.get_token()

class MockInterviewChatRequest(BaseModel):
    message: str
    history: list = []
    topic: str = "General"

@app.post("/mock-interview/chat")
async def mock_interview_chat(request: MockInterviewChatRequest):
    try:
        response = await mock_interview_service.chat(request.message, request.history, request.topic)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ==========================================
# ML Service Endpoints
# ==========================================
from app.services.ml_service import MLService
from fastapi import UploadFile, File, Form

ml_service = MLService()

@app.post("/ml/process_csv")
async def process_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        result = ml_service.process_csv(content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TrainModelRequest(BaseModel):
    # We might need to handle file content differently if persistent, 
    # but for this demo we might need to re-upload or cache.
    # To keep it stateless and simple: Client sends file content or ID?
    # Actually, let's keep it simple: separate endpoints might be tricky without persistence.
    # We'll merge them or require re-upload.
    # For a real app, use S3/Temp storage.
    # Let's assume the client sends the file again OR we use a simple in-memory cache for the session?
    # No, let's make the Client send the file with the train request for simplicity 
    # OR change the architecture to "Upload -> ID -> Train(ID)".
    pass

# Refactoring strategy:
# 1. /ml/upload -> returns (temp_file_id, metadata)
# 2. /ml/train(file_id, config) -> returns results
# 3. /ml/insight(results) -> returns text

# Simple In-Memory Storage for Demo (Not production ready but fast)
import uuid
csv_storage = {}

@app.post("/ml/upload")
async def upload_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        file_id = str(uuid.uuid4())
        csv_storage[file_id] = content # In-memory
        
        # Process meta
        meta = ml_service.process_csv(content)
        return {"file_id": file_id, "metadata": meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TrainRequest(BaseModel):
    file_id: str
    target_column: str
    model_type: str
    task_type: str = "classification"

@app.post("/ml/train")
async def train_model(request: TrainRequest):
    try:
        if request.file_id not in csv_storage:
             raise HTTPException(status_code=404, detail="File session expired or not found")
             
        content = csv_storage[request.file_id]
        result = ml_service.train_model(content, request.target_column, request.model_type, request.task_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class InsightRequest(BaseModel):
    results: dict
    model_type: str

@app.post("/ml/insight")
async def get_insight(request: InsightRequest):
    try:
        insight = await ml_service.generate_insight(request.results, request.model_type)
        return {"insight": insight}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

