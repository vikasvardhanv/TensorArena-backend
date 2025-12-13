import os
import google.generativeai as genai
from typing import List, Dict

class MentorService:
    def __init__(self):
        # Initialize Gemini API
        # Expecting GEMINI_API_KEY to be present in environment variables
        if not os.getenv("GEMINI_API_KEY"):
            print("Warning: GEMINI_API_KEY not found in environment")
            
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    async def chat(self, message: str, history: List[Dict[str, str]]) -> str:
        """
        Conducts a mentoring conversation.
        history format: [{"role": "user" | "model", "parts": ["message"]}]
        """
        
        system_instruction = """
        You are an Adaptive AI Mentor for aspiring AI Engineers.
        Your goal is to TEACH, not just answer. 
        
        When a user asks about a topic (e.g., "Teach me Machine Learning"):
        1. Assess their current knowledge level if not known.
        2. Provide a structured learning path for that topic.
        3. Explain core concepts simply, using analogies.
        4. Suggest study materials (courses, books, papers).
        5. Ask follow-up questions to check understanding.
        
        Do NOT write code solutions for them unless they specifically ask for an example to clear a concept.
        Focus on building intuition and understanding "from scratch".
        """
        
        # Construct the chat session with history
        # Gemini python SDK expects history in format:
        # [{"role": "user", "parts": ["..."]}, {"role": "model", "parts": ["..."]}]
        
        # Prepend system instruction as the start of the interaction context 
        # (Note: true system instructions are set at model creation in some versions, 
        # but for simple chat, we can guide the model via the first prompt or context)
        
        chat_session = self.model.start_chat(history=history)
        
        # Send the new message with the system instruction context if it's the start, 
        # effectively we wrap the user's message with our instruction or rely on the model's persona.
        # For a robust "system" behavior in a chat loop without 'system' role support in 'history',
        # we can inject it.
        
        # However, a cleaner way is to use system_instruction if supported or just prompt engineering.
        # Let's simple prepend the instruction to the very first message if history is empty.
        
        prompt = message
        if not history:
            prompt = f"{system_instruction}\n\nUser: {message}"
            
        response = await chat_session.send_message_async(prompt)
        return response.text
