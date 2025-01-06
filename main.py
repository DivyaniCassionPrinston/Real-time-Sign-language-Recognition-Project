
from real import detect_asl
from train import train_model
import os

# main.py
if __name__ == "__main__":
   
    
    # Get the absolute path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    
    print(f"Looking for data in: {data_dir}")
    
    # First, train the model
    model, classes = train_model(data_dir)
    
    # Then run real-time detection
    detect_asl()