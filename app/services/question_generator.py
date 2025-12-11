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

    async def _generate_with_prompt(self, system_prompt: str, user_prompt: str, imports: str) -> Question:
        if not self.client:
            raise Exception("OpenAI client not initialized")

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
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
                if not template.strip().startswith(imports.strip()): # Use strip to handle potential whitespace differences
                    data["solution_template"] = imports + "\n" + template
            
            return Question(**data)
        except Exception as e:
            print(f"Error generating question: {e}")
            raise Exception(f"Failed to generate question: {str(e)}")

    async def generate_question(self, topic: str, difficulty: str, user_context: Optional[str] = None) -> Question:
        """Standard algorithmic/topic-based question generation."""
        imports = self._get_imports_for_topic(topic)
        
        system_prompts = {
            "Basic": """You are a professor at MIT teaching CS fundamentals. 
            Generate questions that build strong foundations, similar to MIT 6.006.
            Style: MIT OpenCourseWare problem sets.""",
            
            "Intermediate": """You are a Harvard CS50 instructor creating challenging problems.
            Generate questions similar to Harvard CS50 and LeetCode Medium.
            Style: Harvard CS50 assignments with practical scenarios.""",
            
            "Advanced": """You are a competitive programming coach preparing students for IOI/ICPC.
            Generate questions at the level of LeetCode Hard or Codeforces Div1.
            Style: Top-tier competitive programming with rigorous constraints."""
        }

        system_prompt = system_prompts.get(difficulty, system_prompts["Intermediate"])
        
        prompt = f"""
        Create a production-quality coding question for '{topic}' at '{difficulty}' level.
        
        CRITICAL REQUIREMENTS:
        1. **Real-World Context**: Base the problem on actual industry scenarios or research applications.
        2. **Problem Quality**: Clear statement, input/output specs, constraints, examples.
        3. **Test Cases**: 4-6 test cases covering functionality and edge cases.
        4. **Solution Template**: MUST start with imports:\n{imports}\n Then function signature and docstring.
        5. **Answer**: Complete, working, clean Python solution.
        6. **Explanation**: clear, beginner-friendly explanation.

        User Context: {user_context or 'First-time user'}
        
        Return ONLY valid JSON with fields: title, description, difficulty, topic, test_cases, solution_template, answer, explanation.
        """
        
        return await self._generate_with_prompt(system_prompt, prompt, imports)

    async def generate_system_design_question(self, topic: str, difficulty: str) -> Question:
        imports = """# System Design Simulation
from typing import List, Dict, Optional, Set
import time
import random
import collections
"""
        system_prompt = "You are a Principal Engineer at Google designing large-scale distributed systems."
        prompt = f"""
        Create a 'System Design as Code' challenge for topic: '{topic}' ({difficulty}).
        Instead of a diagram, ask the user to implement a Python simulation or skeleton of the system component.
        
        Example topics: Load Balancer, Consistent Hashing, Rate Limiter, KV Store, Distributed ID Generator.
        
        Context: The user needs to write a class or set of functions that simulates the core logic of the system.
        
        Requirements:
        1. **Description**: Explain the system requirements (scalability, latency, availability).
        2. **Task**: Ask user to implement the core algorithm (e.g. "Implement the `get_node(key)` method for consistent hashing").
        3. **Solution Template**: specific class structure with methods to fill in.
        4. **Test Cases**: Scenarios to test the simulation (e.g. "add node, check key distribution").
        
        Return JSON schema matching the standard Question model.
        """
        return await self._generate_with_prompt(system_prompt, prompt, imports)

    async def generate_production_question(self, topic: str, difficulty: str) -> Question:
        imports = """# Production Debugging
import logging
from typing import *
import time
"""
        system_prompt = "You are a Site Reliability Engineer (SRE) handling a production incident."
        prompt = f"""
        Create a debugging challenge for: '{topic}' ({difficulty}).
        Provide a code snippet that has a subtle bug (concurrency issue, memory leak, logic error, off-by-one).
        
        Requirements:
        1. **Scenario**: Describe the production outage or bug report.
        2. **Task**: Fix the bug in the provided code.
        3. **Solution Template**: The BUGGY code that needs fixing.
        4. **Test Cases**: Cases that fail on the buggy code but pass on the fix.
        5. **Answer**: The corrected code.
        
        Return JSON schema matching the standard Question model.
        """
        return await self._generate_with_prompt(system_prompt, prompt, imports)

    async def generate_paper_question(self, topic: str, difficulty: str) -> Question:
        imports = """# Research Implementation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
"""
        system_prompt = "You are an AI Research Scientist at DeepMind implementing state-of-the-art papers."
        prompt = f"""
        Create a challenge to implement a specific mechanism from a famous AI paper related to: '{topic}' ({difficulty}).
        Examples: Multi-Head Attention, LayerNorm, RMSNorm, LoRA adapter, positional encoding.
        
        Requirements:
        1. **Paper Context**: Cite the paper and the specific component.
        2. **Task**: Implement the component from scratch (using PyTorch/NumPy).
        3. **Solution Template**: Class skeleton with `forward` method.
        4. **Answer**: verification of the math/logic.
        
        Return JSON schema matching the standard Question model.
        """
        return await self._generate_with_prompt(system_prompt, prompt, imports)

    async def generate_interview_question(self, topic: str, difficulty: str) -> Question:
        imports = self._get_imports_for_topic("Algorithms")
        system_prompt = "You are a Senior Technical Interviewer at a FAANG company."
        prompt = f"""
        Create a technical interview coding question for: '{topic}' ({difficulty}).
        Focus on communication and problem solving.
        
        Requirements:
        1. **Description**: Interview style problem statement.
        2. **Task**: efficient solution.
        3. **Solution Template**: standard function signature.
        4. **Explanation**: Explain it like you would to an interviewer (trade-offs, time/space complexity).
        
        Return JSON schema matching the standard Question model.
        """
        return await self._generate_with_prompt(system_prompt, prompt, imports)

    async def grade_submission(self, code: str, question_title: str, question_description: str) -> dict:
        """
        Grades the user's coding interview submission.
        """
        if not self.client:
            raise Exception("OpenAI client not initialized")

        system_prompt = """You are a Senior Interviewer at a top tech company (FAANG).
        Your job is to grade a candidate's coding solution.
        Be fair, constructive, and detailed.
        """

        prompt = f"""
        Grade the following Python solution for the interview question: "{question_title}".

        Problem Description:
        {question_description}

        Candidate's Code:
        ```python
        {code}
        ```

        Evaluate on:
        1. **Correctness**: Does it likely solve the problem? (0-10)
        2. **Efficiency**: Time and Space complexity analysis. Is it optimal? (0-10)
        3. **Style**: Variable naming, code structure, readability. (0-10)
        4. **Feedback**: Specific advice on how to improve.

        Return ONLY valid JSON with these fields:
        {{
            "correctness_score": integer (0-10),
            "efficiency_score": integer (0-10),
            "style_score": integer (0-10),
            "feedback": "markdown string with detailed constructive feedback",
            "time_complexity": "string (e.g. O(n))",
            "space_complexity": "string"
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            content = response.choices[0].message.content
            import json
            return json.loads(content)
        except Exception as e:
            print(f"Error grading submission: {e}")
            raise Exception(f"Failed to grade submission: {str(e)}")
