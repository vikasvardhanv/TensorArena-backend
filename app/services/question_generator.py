import os
from typing import List, Optional
from pydantic import BaseModel
import openai

class Question(BaseModel):
    title: str
    description: str
    difficulty: str
    topic: str
    test_cases: List[str]
    solution_template: str

class QuestionGenerator:
    def __init__(self):
        # In a real app, use env vars. For now, we'll check if it's set.
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            self.client = None

    async def generate_question(self, topic: str, difficulty: str, user_context: Optional[str] = None) -> Question:
        """
        Generates a coding question based on the topic and difficulty.
        """
        if not self.client:
            # Mock response if no API key
            return self._get_mock_question(topic, difficulty)

        system_prompt = "You are an expert AI coding tutor."
        if difficulty == "Advanced":
            system_prompt = """You are a ruthless competitive programming setter (like for Codeforces or LeetCode Hard). 
            Generate extremely difficult, complex problems that require advanced algorithms (DP, Graphs, Segment Trees, etc.).
            Focus on edge cases, strict time/space constraints, and non-obvious solutions."""

        prompt = f"""
        Create a coding question for the topic '{topic}' with difficulty '{difficulty}'.
        User Context: {user_context or 'None'}
        
        Requirements:
        - Title: Catchy and professional.
        - Description: Detailed, rigorous, with mathematical notation if needed. Include Input/Output format and Constraints.
        - Difficulty: {difficulty} (If Advanced, make it 'Crazy Hard').
        - Test Cases: Provide 3-5 distinct test cases (including edge cases).
        - Solution Template: Python function signature.
        
        Return the response in JSON format with the following fields:
        - title
        - description (markdown)
        - difficulty
        - topic
        - test_cases (list of strings)
        - solution_template
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            # Parse JSON (assuming the model returns valid JSON as requested)
            import json
            data = json.loads(content)
            return Question(**data)
        except Exception as e:
            print(f"Error generating question: {e}")
            return self._get_mock_question(topic, difficulty)

    def _get_mock_question(self, topic: str, difficulty: str) -> Question:
        return Question(
            title=f"Sample {topic} Problem",
            description=f"This is a sample problem for {topic} at {difficulty} level. (API Key missing)",
            difficulty=difficulty,
            topic=topic,
            test_cases=["assert solve(1) == 1"],
            solution_template="def solve(n):\n    pass"
        )
