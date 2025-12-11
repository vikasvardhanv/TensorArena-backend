import os
import json
from typing import List, Optional
from pydantic import BaseModel
import openai


class RoleBasedQuestion(BaseModel):
    title: str
    scenario: str
    type: str  # "multiple-choice", "fill-in-blank", "output-selection", "coding"
    role: str
    options: Optional[List[str]] = None
    correctAnswer: Optional[int | str] = None
    explanation: Optional[str] = None
    codeSnippet: Optional[str] = None
    expectedOutput: Optional[str] = None


class RoleBasedQuestionGenerator:
    """
    Generates high-quality, scenario-based questions tailored to specific AI/ML roles.
    Questions are designed to test practical, real-world knowledge and decision-making.
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            raise Exception("OpenAI API key is required for question generation")

        # Role-specific context and focus areas
        self.role_contexts = {
            "Machine Learning Engineer": {
                "focus": "production ML systems, model deployment, MLOps, data pipelines, model monitoring",
                "scenarios": [
                    "Production model serving issues",
                    "Feature engineering pipeline bugs",
                    "Model retraining strategies",
                    "A/B testing and experimentation",
                    "Data drift detection",
                    "Model performance degradation"
                ],
                "skills": "TensorFlow, PyTorch, Kubernetes, Docker, ML pipelines, CI/CD for ML"
            },
            "Data Scientist": {
                "focus": "statistical analysis, experimentation, model selection, business impact",
                "scenarios": [
                    "Experimental design for product features",
                    "Statistical significance in A/B tests",
                    "Model selection and validation",
                    "Handling imbalanced datasets",
                    "Feature importance analysis",
                    "Communicating results to stakeholders"
                ],
                "skills": "Statistics, Python, R, SQL, visualization, hypothesis testing"
            },
            "Computer Vision Engineer": {
                "focus": "image processing, object detection, segmentation, video analysis",
                "scenarios": [
                    "Real-time video processing optimization",
                    "Object detection model debugging",
                    "Image preprocessing pipeline issues",
                    "Multi-camera calibration",
                    "Edge deployment constraints",
                    "Handling varying lighting conditions"
                ],
                "skills": "OpenCV, YOLO, CNNs, image augmentation, GPU optimization"
            },
            "NLP Engineer": {
                "focus": "text processing, language models, embeddings, information extraction",
                "scenarios": [
                    "Named Entity Recognition failures",
                    "Sentiment analysis bias",
                    "Text classification in production",
                    "Tokenization edge cases",
                    "Multilingual model deployment",
                    "Context window limitations"
                ],
                "skills": "Transformers, BERT, spaCy, tokenization, embeddings, fine-tuning"
            },
            "LLM Specialist": {
                "focus": "prompt engineering, LLM fine-tuning, RAG systems, LLM deployment",
                "scenarios": [
                    "Prompt injection attacks",
                    "RAG system retrieval quality",
                    "LLM hallucination mitigation",
                    "Context window management",
                    "Fine-tuning for domain adaptation",
                    "Inference cost optimization"
                ],
                "skills": "GPT, Claude, LangChain, vector databases, prompt optimization, RLHF"
            },
            "AI Product Manager": {
                "focus": "product strategy, ML feasibility, user experience, ROI analysis",
                "scenarios": [
                    "ML feature prioritization",
                    "Model performance vs user experience tradeoffs",
                    "Data privacy and compliance",
                    "Cold start problems in recommendations",
                    "ML project scoping and timelines",
                    "Stakeholder communication about model limitations"
                ],
                "skills": "Product strategy, ML basics, metrics, UX, business impact"
            }
        }

    def _get_question_type_distribution(self, count: int) -> List[str]:
        """
        Distribute question types across the requested count.
        Ensures diversity in question formats.
        """
        types = ["multiple-choice", "output-selection", "fill-in-blank"]

        # For 3 questions: 1 of each type
        # For more: distribute evenly with preference for multiple-choice
        if count <= 3:
            return types[:count]

        distribution = []
        base_count = count // 3
        remainder = count % 3

        for i, qtype in enumerate(types):
            type_count = base_count + (1 if i < remainder else 0)
            distribution.extend([qtype] * type_count)

        return distribution

    async def generate_role_questions(self, role: str, count: int = 3) -> List[RoleBasedQuestion]:
        """
        Generate multiple high-quality, scenario-based questions for a specific role.
        """
        if role not in self.role_contexts:
            raise ValueError(f"Unknown role: {role}. Available roles: {list(self.role_contexts.keys())}")

        if not self.client:
            raise Exception("OpenAI client not initialized")

        role_context = self.role_contexts[role]
        question_types = self._get_question_type_distribution(count)

        # System prompt emphasizing quality and realism
        system_prompt = f"""You are a senior {role} interviewer at a top tech company (Google, Meta, OpenAI, Anthropic).
You create REALISTIC, SCENARIO-BASED technical questions that test PRACTICAL KNOWLEDGE and DECISION-MAKING.

Your questions should:
1. Be based on REAL production scenarios and incidents
2. Test practical understanding, not just theoretical knowledge
3. Include specific technical details and context
4. Have clear, unambiguous correct answers
5. Provide educational explanations that teach best practices

Focus areas for {role}:
- {role_context['focus']}
- Key skills: {role_context['skills']}

Common scenarios to draw from:
{chr(10).join('- ' + s for s in role_context['scenarios'])}"""

        # User prompt for generating all questions at once
        user_prompt = f"""Generate {count} distinct scenario-based questions for a {role}.

CRITICAL REQUIREMENTS:

1. **Scenario Realism**: Each question must present a REAL production scenario with:
   - Specific technical context (versions, metrics, error messages)
   - Business impact or urgency
   - Concrete symptoms or observations
   - Realistic constraints (time, resources, users affected)

2. **Question Types Distribution**:
{chr(10).join(f'   - Question {i+1}: {qtype}' for i, qtype in enumerate(question_types))}

3. **Question Type Formats**:

   **multiple-choice**:
   - Present a production scenario/incident
   - Provide 4 realistic options (all should be plausible)
   - correctAnswer: index (0-3) of the correct option
   - Options should test understanding of trade-offs and best practices

   **output-selection**:
   - Provide a code snippet relevant to the role
   - Ask what the output/behavior will be
   - Provide 4 possible outputs as options
   - correctAnswer: index (0-3) of the correct output
   - Include edge cases or common pitfalls

   **fill-in-blank**:
   - Present a scenario with a specific technical question
   - correctAnswer: the exact term, command, or value (as a string)
   - Should test specific technical knowledge

4. **Code Snippets** (for output-selection and some multiple-choice):
   - Use realistic code from production scenarios
   - Include common bugs or pitfalls
   - Keep code concise (10-30 lines max)
   - Use proper syntax and realistic variable names

5. **Explanations**:
   - Explain WHY the correct answer is right
   - Explain WHY other options are wrong or suboptimal
   - Include best practices and lessons learned
   - Keep concise but educational (3-5 sentences)

6. **Difficulty Balance**:
   - Mix easy, medium, and hard questions
   - Test both breadth (different scenarios) and depth (technical details)
   - Include recent/current technology and practices

EXAMPLE SCENARIOS BY ROLE:

Machine Learning Engineer:
- "Your model's inference latency jumped from 50ms to 800ms after deployment. CPU at 30%, GPU memory at 95%..."
- "After retraining your recommendation model, you notice AUC improved from 0.85 to 0.92, but user CTR dropped 15%..."

Data Scientist:
- "Your A/B test shows 8% improvement (p=0.04) with 1000 users per variant. Should you ship?"
- "You notice your churn prediction model has 95% accuracy but only 5% precision on churners..."

Computer Vision Engineer:
- "Your object detection model runs at 30 FPS on your dev machine but 2 FPS on the edge device..."
- "Your face recognition system works perfectly in the office but fails in production outdoor settings..."

NLP Engineer:
- "Your NER model correctly identifies 'Apple' as ORG in 'Apple released iPhone' but tags it as FRUIT in other contexts..."
- "Users are getting offensive completions from your text generation model despite filtering..."

LLM Specialist:
- "Your RAG system retrieves relevant docs but the LLM still hallucinates specific numbers..."
- "After increasing context window from 4k to 32k tokens, your response quality degraded..."

AI Product Manager:
- "Engineering says the ML model needs 6 months and 2M labeled examples. Marketing wants it in Q1..."
- "Your personalization feature improves engagement 25% but increases server costs 300%..."

Return ONLY valid JSON in the following format (must be an object with "questions" key):
{{
    "questions": [
        {{
            "title": "Concise, professional title",
            "scenario": "Detailed scenario with specific technical context",
            "type": "multiple-choice|output-selection|fill-in-blank",
            "role": "{role}",
            "options": ["option1", "option2", "option3", "option4"],
            "correctAnswer": 0,
            "explanation": "Clear explanation of why the answer is correct and others are wrong",
            "codeSnippet": "Python code if applicable"
        }}
    ]
}}

IMPORTANT:
- Make questions SPECIFIC and REALISTIC
- Avoid generic or theoretical questions
- Include actual error messages, metrics, or observations
- Test decision-making under realistic constraints
- Ensure explanations are educational and actionable"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4 for higher quality questions
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.9  # Higher temperature for more diverse scenarios
            )

            content = response.choices[0].message.content
            data = json.loads(content)

            # Handle both array and object responses
            if isinstance(data, dict) and "questions" in data:
                questions_data = data["questions"]
            elif isinstance(data, list):
                questions_data = data
            else:
                # Provide detailed error for debugging
                available_keys = list(data.keys()) if isinstance(data, dict) else "not a dict"
                error_msg = f"Unexpected response format from LLM. Expected 'questions' key in object or array. Got: {type(data).__name__}"
                if isinstance(data, dict):
                    error_msg += f" with keys: {available_keys}"
                print(f"DEBUG: {error_msg}")
                print(f"DEBUG: Response content: {content[:500]}")  # First 500 chars
                raise ValueError(error_msg)

            # Validate and parse questions
            questions = []
            for q_data in questions_data:
                # Ensure role is set
                q_data["role"] = role

                # Convert correctAnswer to appropriate type
                if q_data["type"] == "fill-in-blank":
                    q_data["correctAnswer"] = str(q_data["correctAnswer"])
                else:
                    q_data["correctAnswer"] = int(q_data["correctAnswer"])

                questions.append(RoleBasedQuestion(**q_data))

            return questions[:count]  # Ensure we return exactly count questions

        except Exception as e:
            print(f"Error generating role-based questions: {e}")
            raise Exception(f"Failed to generate role-based questions: {str(e)}")

    async def generate_single_question(self, role: str, question_type: str) -> RoleBasedQuestion:
        """
        Generate a single role-based question of a specific type.
        Useful for regenerating or adding individual questions.
        """
        questions = await self.generate_role_questions(role, count=1)
        return questions[0]
