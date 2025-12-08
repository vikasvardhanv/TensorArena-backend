import subprocess
import tempfile
import os
from typing import Dict

class CodeExecutor:
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
    
    def execute_python(self, code: str) -> Dict[str, str]:
        """
        Execute Python code safely in a subprocess with timeout.
        Returns dict with 'output' and 'error' keys.
        """
        try:
            # Create a temporary file to store the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Execute the code in a subprocess with timeout
                result = subprocess.run(
                    ['python3', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                if result.returncode == 0:
                    return {
                        "output": result.stdout or "Code executed successfully (no output)",
                        "error": None
                    }
                else:
                    return {
                        "output": result.stdout,
                        "error": result.stderr or "Execution failed"
                    }
            finally:
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    
        except subprocess.TimeoutExpired:
            return {
                "output": "",
                "error": f"Execution timed out after {self.timeout} seconds"
            }
        except Exception as e:
            return {
                "output": "",
                "error": f"Execution error: {str(e)}"
            }
