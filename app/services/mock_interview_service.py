
import os
import google.generativeai as genai
from typing import List, Dict

class MockInterviewService:
    def __init__(self):
        if not os.getenv("GEMINI_API_KEY"):
            print("Warning: GEMINI_API_KEY not found in environment")
            
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        self.system_instruction = """
        You are an expert technical interviewer at a top tech company (FAANG level).
        Your goal is to conduct a realistic mock interview.
        
        Guidelines:
        - Be professional but encouraging.
        - Ask one clear question at a time.
        - If the user's answer is good, praise them briefly and move to a deeper follow-up or a new topic.
        - If the user's answer is vague, ask for clarification.
        - If the user is stuck, provide a small hint.
        - Keep your responses concise (spoken length) so the user doesn't get bored listening.
        - Start by asking what topic they want to cover if not provided.
        - Topics can be System Design, Algorithms (Conceptual), Behavioral, or Specific Technologies.
        
        The user inputs will come from speech-to-text, so expect some informality or transcription errors. Infer intent.
        """

    async def chat(self, message: str, history: List[Dict[str, str]], topic: str = "General") -> str:
        """
        Generates a response to the user's message/answer during the interview.
        """
        try:
            chat_session = self.model.start_chat(history=[
                {"role": "user", "parts": [f"System Instruction: {self.system_instruction}"]},
                {"role": "model", "parts": ["Understood. I am ready to conduct the interview."]}
            ] + [
                {"role": h["role"], "parts": [h["text"]]} for h in history
            ])
            
            # Contextualize with topic if starting
            msg = message
            if not history and topic:
                msg = f"I want to interview for {topic}. {message}"
                
            response = chat_session.send_message(msg)
            return response.text
        except Exception as e:
            print(f"Error in MockInterviewService: {e}")
            return "I'm having trouble connecting to my brain right now. Please try again."
