# real_time_detection.py
import cv2
import torch
import numpy as np
from torchvision import transforms
from model import ASLNet

def detect_asl():
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASLNet().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Define the classes
    classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
              'U', 'V', 'W', 'X', 'Y', 'Z']
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(frame_rgb)
        input_batch = input_tensor.unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
            
        # Draw the prediction on the frame
        prediction = classes[predicted_idx]
        cv2.putText(frame, f'Prediction: {prediction} ({confidence:.2f})',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('ASL Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

