from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.services.question_generator import QuestionGenerator
from app.services.code_executor import CodeExecutor
from app.services.role_question_generator import RoleBasedQuestionGenerator

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
