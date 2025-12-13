import os
import google.generativeai as genai
from typing import List, Dict

class SystemDesignService:
    def __init__(self):
        if not os.getenv("GEMINI_API_KEY"):
            print("Warning: GEMINI_API_KEY not found in environment")
            
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    async def chat(self, message: str, history: List[Dict[str, str]], topic: str = "General System Design") -> str:
        """
        Conducts a System Design interview/session.
        """
        
        system_instruction = f"""
        You are a Staff Technical Interviewer at a top tech company, specializing in {topic}.
        Your goal is to conduct a collaborative system design session with the user.
        
        Context: The user wants to design: {topic}
        
        Guidelines:
        1. **Requirements Gathering**: If the user just started, ask for functional and non-functional requirements (throughput, latency, scale).
        2. **High-Level Design**: Guide them to define the core components (Load Balancers, API Gateway, DBs, Queues).
        3. **Deep Dive**: Drill down into specific areas relevant to AI Infrastructure (e.g., Sharding strategies for LLMs, KV caching for Inference).
        4. **Trade-offs**: Challenge their decisions. "Why Redis over Memcached here?", "How do you handle node failure during training?"
        
        Keep responses concise (under 200 words) to encourage back-and-forth dialogue.
        Use Markdown for structure.
        """
        
        chat_session = self.model.start_chat(history=history)
        
        prompt = message
        if not history:
            prompt = f"{system_instruction}\n\nCandidate: {message}"
            
        response = await chat_session.send_message_async(prompt)
        return response.text
