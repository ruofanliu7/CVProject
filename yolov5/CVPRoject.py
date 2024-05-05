import torch
from torchvision import transforms
from PIL import Image
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

def detect_objects(image):
    # Perform inference
    results = model(image)

    # Process results
    detections = results[0]  # Assuming detections are in the first element
    objects = []

    for det in detections:
        # Extract relevant info
        x1, y1, x2, y2, conf, cls = det[:6].tolist()
        
        # Check if the class corresponds to a sports ball (assuming class ID 0 for sports ball)
        if int(cls) == 0:
            objects.append((x1, y1, x2, y2, conf, cls))

    return objects


def main():
    video_path = 'A:\VSCode/CVProject/yolov5/FIFAVideo.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Detect objects in the frame
        objects = detect_objects(image)

        # Draw bounding boxes on the frame
        for obj in objects:
            x1, y1, x2, y2, conf, cls = obj  # Unpack the tuple directly
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f'Class: {int(cls)}, Confidence: {conf:.2f}', (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the frame with bounding boxes
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
