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
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error

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
            df = pd.read_csv(BytesIO(file_content), nrows=5000)
            
            # Basic cleaning: Drop NaNs for simplicity in this demo
            df = df.dropna()
            
            # Limit rows for performance if needed (already limited by read_csv)

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
            
            # Determine task type if not explicitly correct (simple heuristic based on user input or target)
            # For now, we trust the model_type mapping, but we set a flag
            is_regression = model_type in [
                "linear_regression", "ridge", "lasso", "svr", 
                "random_forest_regressor", "gradient_boosting_regressor", 
                "mlp_regressor", "knn_regressor", "decision_tree_regressor", "adaboost_regressor"
            ]
            
            model = None
            # Classification Models
            if model_type == "logistic_regression": model = LogisticRegression(max_iter=1000)
            elif model_type == "random_forest": model = RandomForestClassifier()
            elif model_type == "svm": model = SVC()
            elif model_type == "knn": model = KNeighborsClassifier()
            elif model_type == "gradient_boosting": model = GradientBoostingClassifier()
            elif model_type == "decision_tree": model = DecisionTreeClassifier()
            elif model_type == "naive_bayes": model = GaussianNB()
            elif model_type == "mlp_classifier": model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
            elif model_type == "adaboost": model = AdaBoostClassifier()
            
            # Regression Models
            elif model_type == "linear_regression": model = LinearRegression()
            elif model_type == "ridge": model = Ridge()
            elif model_type == "lasso": model = Lasso()
            elif model_type == "svr": model = SVR()
            elif model_type == "random_forest_regressor": model = RandomForestRegressor()
            elif model_type == "gradient_boosting_regressor": model = GradientBoostingRegressor()
            elif model_type == "mlp_regressor": model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500)
            elif model_type == "knn_regressor": model = KNeighborsRegressor()
            elif model_type == "decision_tree_regressor": model = DecisionTreeRegressor()
            elif model_type == "adaboost_regressor": model = AdaBoostRegressor()
            
            # Clustering
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
            
            result_metrics = {}
            if is_regression:
                 mse = mean_squared_error(y_test, y_pred)
                 r2 = r2_score(y_test, y_pred)
                 mae = mean_absolute_error(y_test, y_pred)
                 result_metrics = {"mse": mse, "r2": r2, "mae": mae}
            else:
                # Classification Metrics
                acc = accuracy_score(y_test, y_pred)
                # Handle potential issue if classes don't match
                try:
                    report = classification_report(y_test, y_pred, output_dict=True)
                except:
                    report = {"error": "Could not generate report (class mismatch?)"}
                cm = confusion_matrix(y_test, y_pred).tolist()
                result_metrics = {"accuracy": acc, "report": report, "confusion_matrix": cm}
            
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
                # Same logic for top features, but for regression "Sample" might be Actual vs Predicted?
                # For consistency in this "Playground", we'll keep the Scatter Plot of features
                # But for Regression, a "Predicted vs Actual" plot is often better.
                # Let's stick to Feature Space for now to keep the frontend simple, 
                # or add a specific regression plot logic.
                
                # 1. Identify Top 2 Features
                sorted_features = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
                if not sorted_features:
                    top_features = X.columns[:2].tolist()
                else:
                    top_features = [k for k, v in sorted_features[:2]]
                
                sample_size = min(200, len(X_test))
                viz_data = []
                
                # Reset index 
                X_test_sample = X_test[top_features].iloc[:sample_size].reset_index(drop=True)
                y_test_sample = y_test.iloc[:sample_size].reset_index(drop=True)
                y_pred_sample = y_pred[:sample_size]
                
                for i in range(sample_size):
                    val_x = float(X_test_sample.iloc[i][0])
                    val_y = float(X_test_sample.iloc[i][1])
                    
                    # Convert labels/predictions safely
                    if is_regression:
                        true_label = float(y_test_sample.iloc[i])
                        pred_label = float(y_pred_sample[i])
                    else:
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
                    "data": viz_data,
                    "type": "regression" if is_regression else "classification" 
                }

            return {
                "metrics": result_metrics,
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
            
        # Optimize prompt: Limit feature importance to top 20 to save tokens
        feature_imp = results.get('feature_importance', {})
        top_features = dict(sorted(feature_imp.items(), key=lambda item: item[1], reverse=True)[:20])

        prompt = f"""
        You are an expert Machine Learning Engineer.
        Analyze the following training results for a {model_type} model:
        
        Accuracy: {results.get('metrics', {}).get('accuracy', 'N/A')}
        Top 20 Feature Importance: {json.dumps(top_features)}
        Confusion Matrix: {json.dumps(results.get('metrics', {}).get('confusion_matrix', []))}
        
        Provide a very concise 3-bullet point summary (fast response needed):
        1. Model Performance check (Good/Bad?)
        2. Top driving features (What matters most?)
        3. One recommendation to improve.
        """
        
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            return f"Failed to generate insight: {str(e)}"
