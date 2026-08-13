import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def train_model():
    print("Generating synthetic dataset...")
    # Create a synthetic dataset for fraud detection
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_informative=8, 
        n_redundant=2, 
        random_state=42
    )
    
    print("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model in the same directory as this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model.pkl")
    
    print(f"Saving model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    train_model()
