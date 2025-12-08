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
    answer: str
    explanation: str

class QuestionGenerator:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            raise Exception("OpenAI API key is required for question generation")

    def _get_imports_for_topic(self, topic: str) -> str:
        """Get appropriate imports and setup based on topic"""
        imports_map = {
            "Python": """# Python Standard Library
from typing import List, Dict, Set, Tuple, Optional
import math
import collections
""",
            "Data Structures": """# Data Structures
from typing import List, Dict, Set, Tuple, Optional
import collections
from collections import deque, defaultdict, Counter
import heapq
""",
            "Algorithms": """# Algorithms
from typing import List, Dict, Set, Tuple, Optional
import math
import bisect
from collections import deque, defaultdict
import heapq
""",
            "Machine Learning": """# Machine Learning
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
# Note: Install required packages: pip install numpy pandas scikit-learn
""",
            "Neural Networks": """# Neural Networks / Deep Learning
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# Note: Install required packages: pip install numpy torch
"""
        }
        return imports_map.get(topic, "# Python\nfrom typing import List, Dict, Optional\n")

    async def generate_question(self, topic: str, difficulty: str, user_context: Optional[str] = None) -> Question:
        """
        Generates MIT/Harvard/LeetCode-level coding questions with proper imports.
        """
        if not self.client:
            raise Exception("OpenAI client not initialized")

        # Get appropriate imports for the topic
        imports = self._get_imports_for_topic(topic)

        # Enhanced system prompts based on difficulty
        system_prompts = {
            "Basic": """You are a professor at MIT teaching CS fundamentals. 
            Generate questions that build strong foundations, similar to MIT 6.006 (Introduction to Algorithms).
            Focus on: clarity, proper problem formulation, and teaching core concepts.
            Style: MIT OpenCourseWare problem sets.""",
            
            "Intermediate": """You are a Harvard CS50 instructor creating challenging problems.
            Generate questions similar to Harvard CS50 problem sets and LeetCode Medium.
            Focus on: real-world applications, data structure manipulation, and algorithmic thinking.
            Style: Harvard CS50 assignments with practical scenarios.""",
            
            "Advanced": """You are a competitive programming coach preparing students for IOI/ICPC.
            Generate questions at the level of LeetCode Hard, Codeforces Div1, or MIT 6.046 (Advanced Algorithms).
            Focus on: complex algorithms (DP, Graph Theory, Segment Trees), optimization, and edge cases.
            Style: Top-tier competitive programming with rigorous constraints."""
        }

        system_prompt = system_prompts.get(difficulty, system_prompts["Intermediate"])

        # Enhanced prompt with real-world context
        prompt = f"""
        Create a production-quality coding question for '{topic}' at '{difficulty}' level.
        
        CRITICAL REQUIREMENTS:
        1. **Real-World Context**: Base the problem on actual industry scenarios or research applications
           - For ML/AI: Use cases from papers, production systems, or research labs
           - For Algorithms: Practical applications (routing, scheduling, optimization)
           - For Data Structures: System design scenarios (caching, databases, networks)
        
        2. **Problem Quality Standards**:
           - Title: Professional and descriptive (like LeetCode)
           - Description: 
             * Clear problem statement with motivation
             * Precise input/output specifications
             * Constraints (time/space complexity expectations)
             * Examples with explanations
             * Edge cases to consider
        
        3. **Difficulty Calibration**:
           - Basic: MIT 6.006 level - foundational concepts, clear solutions
           - Intermediate: Harvard CS50/LeetCode Medium - requires insight and optimization
           - Advanced: LeetCode Hard/Codeforces - multiple techniques, non-obvious solutions
        
        4. **Test Cases**: 
           - Include 4-6 test cases covering:
             * Basic functionality
             * Edge cases (empty, single element, maximum size)
             * Corner cases specific to the algorithm
             * Performance test (if Advanced)
        
        5. **Solution Template**: 
           - MUST start with these exact imports (DO NOT modify):
{imports}
           - Then provide a Python function signature with type hints
           - Include a detailed docstring with:
             * Problem description
             * Parameter descriptions with types
             * Return value description
             * Example usage
             * Time/Space complexity notes
           - Add clear comments:
             * "# TODO: Write your solution here" in the function body
             * "# Example: result = your_function(test_input)" after the function
             * "# To test: Click 'Run Code' button above" at the bottom
           - Make it beginner-friendly with helpful hints
           
        6. **Answer and Explanation**:
           - Provide the BEST and SIMPLEST solution for the problem.
           - Provide a very simple, easy-to-understand explanation of the logic.
           - Avoid complex theory; explain it like you would to a beginner.
        
        User Context: {user_context or 'First-time user'}
        
        Return ONLY valid JSON with these exact fields:
        {{
            "title": "string",
            "description": "markdown string with ## headers",
            "difficulty": "{difficulty}",
            "topic": "{topic}",
            "test_cases": ["list", "of", "strings"],
            "solution_template": "python code string starting with imports",
            "answer": "complete python solution code",
            "explanation": "simple markdown string explaining the solution"
        }}
        
        IMPORTANT: The solution_template MUST include the imports provided above at the start.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.8
            )
            
            content = response.choices[0].message.content
            import json
            data = json.loads(content)
            
            # Ensure imports are at the start of solution template
            if "solution_template" in data:
                template = data["solution_template"]
                if not template.startswith(imports.strip()):
                    data["solution_template"] = imports + "\n" + template
            
            return Question(**data)
        except Exception as e:
            print(f"Error generating question: {e}")
            raise Exception(f"Failed to generate question: {str(e)}")
