from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch, MagicMock

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the AI LeetCode Platform API"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@patch("app.services.question_generator.QuestionGenerator.generate_question")
def test_generate_question(mock_generate):
    # Mock the async method
    async def mock_return(*args, **kwargs):
        return {
            "title": "Test Question",
            "description": "Test Description",
            "difficulty": "Basic",
            "topic": "Python",
            "test_cases": [],
            "solution_template": "def solve(): pass"
        }
    mock_generate.side_effect = mock_return

    response = client.post("/generate_question", json={
        "topic": "Python",
        "difficulty": "Basic"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Test Question"
