import io
import cv2
import numpy as np
from PIL import Image
from fastapi.responses import Response

def read_image_file(file_content) -> np.ndarray:
    # Convert the uploaded file to an image
    image = Image.open(io.BytesIO(file_content))
    # Convert PIL Image to cv2 image
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image

def create_image_response(predictions: list) -> Response:
    # Create a copy of the original image to draw on
    image = cv2.imread("temp_image.jpg")  # We need to save the original image temporarily
    
    # Draw bounding boxes and labels for each prediction
    for pred in predictions:
        # Get box coordinates
        x1 = int(pred["box"]["x1"])
        y1 = int(pred["box"]["y1"])
        x2 = int(pred["box"]["x2"])
        y2 = int(pred["box"]["y2"])
        
        # Get class name and confidence score
        class_name = pred["class"]
        score = pred["score"]
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare label text
        label = f"{class_name}: {score:.2f}"
        
        # Get label size for background rectangle
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw label background
        cv2.rectangle(
            image,
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width, y1),
            (0, 255, 0),
            -1,
        )
        
        # Draw label text
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
    
    # Convert the image to bytes
    _, img_encoded = cv2.imencode('.jpg', image)
    
    # Return the image as a response
    return Response(content=img_encoded.tobytes(), media_type="image/jpeg")
    