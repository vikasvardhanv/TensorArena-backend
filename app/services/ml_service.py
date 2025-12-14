import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import os
import google.generativeai as genai

class MLService:
    def __init__(self):
        if os.getenv("GEMINI_API_KEY"):
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            print("Warning: GEMINI_API_KEY not found")
            self.model = None

    def process_csv(self, file_content: bytes) -> dict:
        """
        Reads CSV bytes and returns metadata (columns, head, shape).
        """
        try:
            # Create a temporary file-like object
            from io import BytesIO
            df = pd.read_csv(BytesIO(file_content))
            
            # Basic cleaning: Drop NaNs for simplicity in this demo
            df = df.dropna()
            
            # Limit rows for performance if needed (e.g. max 2000 for rapid demo)
            if len(df) > 5000:
                df = df.head(5000)

            return {
                "columns": df.columns.tolist(),
                "head": df.head(5).to_dict(orient="records"),
                "shape": df.shape,
                "dtypes": {k: str(v) for k, v in df.dtypes.items()},
                "summary": df.describe().to_dict()
            }
        except Exception as e:
            return {"error": str(e)}

    def train_model(self, file_content: bytes, target_column: str, model_type: str, task_type: str = "classification"):
        try:
            from io import BytesIO
            df = pd.read_csv(BytesIO(file_content)).dropna()
            
            if target_column not in df.columns:
                return {"error": f"Target column '{target_column}' not found"}

            X = df.drop(columns=[target_column])
            y = df[target_column]

            # --- Data Cleaning: Remove ID-like columns ---
            # 1. Drop columns with "id" in their name (case-insensitive)
            cols_to_drop = [c for c in X.columns if 'id' in c.lower()]
            # 2. Drop high-cardinality string columns (likely identifiers)
            for col in X.select_dtypes(include=['object', 'string']).columns:
                if col not in cols_to_drop:
                    # If unique count is > 90% of row count, it's likely an ID
                    if X[col].nunique() > 0.9 * len(X):
                        cols_to_drop.append(col)
            
            if cols_to_drop:
                # print(f"Dropping ID-like columns: {cols_to_drop}") # Logging for debug
                X = X.drop(columns=cols_to_drop)

            # Simple encoding for categorical features if any
            X = pd.get_dummies(X, drop_first=True)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = None
            if model_type == "logistic_regression":
                model = LogisticRegression(max_iter=1000)
            elif model_type == "random_forest":
                model = RandomForestClassifier()
            elif model_type == "svm":
                model = SVC()
            elif model_type == "knn":
                model = KNeighborsClassifier()
            elif model_type == "gradient_boosting":
                model = GradientBoostingClassifier()
            elif model_type == "decision_tree":
                model = DecisionTreeClassifier()
            elif model_type == "naive_bayes":
                model = GaussianNB()
            elif model_type == "kmeans":
                # Unsupervised
                model = KMeans(n_clusters=3) # Defaulting to 3 for demo
                model.fit(X)
                return {
                    "metrics": {"inertia": float(model.inertia_)},
                    "model_type": model_type,
                    "feature_names": X.columns.tolist()
                }
            else:
                return {"error": f"Unsupported model type: {model_type}"}

            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred).tolist()
            
            # Feature Importance (if applicable)
            feature_importance = {}
            if hasattr(model, "feature_importances_"):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
            elif hasattr(model, "coef_"):
                # For linear models, take abs of coefs
                if model.coef_.ndim > 1:
                     feature_importance = dict(zip(X.columns, np.abs(model.coef_[0])))
                else:
                     feature_importance = dict(zip(X.columns, np.abs(model.coef_)))
            
            # --- Visualization Data Preparation ---
            visualization = {}
            if len(X.columns) >= 2:
                # 1. Identify Top 2 Features
                sorted_features = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
                # Fallback if feature importance is empty or all zeros (rare but possible)
                if not sorted_features:
                    top_features = X.columns[:2].tolist()
                else:
                    top_features = [k for k, v in sorted_features[:2]]
                
                # 2. Sample Data for Plotting (max 200 points from Test set)
                # We reuse X_test and y_test, and y_pred
                sample_size = min(200, len(X_test))
                
                # We need to ensure we are using the original values for plotting, not dummies if possible
                # But here X is already encoded. We'll plot the encoded values which is correct for the model.
                
                # Convert to list of dicts
                viz_data = []
                # Reset index to make iteration easy
                X_test_sample = X_test[top_features].iloc[:sample_size].reset_index(drop=True)
                y_test_sample = y_test.iloc[:sample_size].reset_index(drop=True)
                y_pred_sample = y_pred[:sample_size] # y_pred is already numpy array
                
                for i in range(sample_size):
                    val_x = float(X_test_sample.iloc[i][0])
                    val_y = float(X_test_sample.iloc[i][1])
                    true_label = int(y_test_sample.iloc[i]) if isinstance(y_test_sample.iloc[i], (int, np.integer)) else str(y_test_sample.iloc[i])
                    pred_label = int(y_pred_sample[i]) if isinstance(y_pred_sample[i], (int, np.integer)) else str(y_pred_sample[i])
                    
                    viz_data.append({
                        "x": val_x,
                        "y": val_y,
                        "label": true_label,
                        "prediction": pred_label
                    })

                visualization = {
                    "features": top_features,
                    "data": viz_data
                }

            return {
                "metrics": {
                    "accuracy": acc,
                    "report": report,
                    "confusion_matrix": cm
                },
                "feature_importance": feature_importance,
                "visualization": visualization,
                "model_type": model_type
            }

        except Exception as e:
            return {"error": str(e)}

    async def generate_insight(self, results: dict, model_type: str) -> str:
        """
        Uses Gemini to explain the results.
        """
        if not self.model:
            return "LLM service not available."
            
        prompt = f"""
        You are an expert Machine Learning Engineer.
        Analyze the following training results for a {model_type} model:
        
        Accuracy: {results.get('metrics', {}).get('accuracy', 'N/A')}
        Feature Importance: {json.dumps(results.get('feature_importance', {}))}
        Confusion Matrix: {json.dumps(results.get('metrics', {}).get('confusion_matrix', []))}
        
        Provide a concise 3-bullet point summary:
        1. Model Performance check (Good/Bad?)
        2. Top driving features (What matters most?)
        3. One recommendation to improve.
        """
        
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            return f"Failed to generate insight: {str(e)}"
