import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import cv2

def corners(image: np.ndarray, model_path: str, target_size=(224, 224)) -> np.ndarray:
    # Load the trained model
    model = load_model(model_path)
    
    # Preprocess the image: resize and normalize
    resized_image = cv2.resize(image, target_size)
    normalized_image = resized_image / 255.0  # Normalize pixel values to [0, 1]
    
    # Expand dimensions to fit the model's input shape (batch size)
    input_image = np.expand_dims(normalized_image, axis=0)
    
    # Use the model to predict the corners
    predicted_points = model.predict(input_image)
    
    # Denormalize the predicted points to match the original image size
    original_height, original_width, _ = image.shape
    new_height, new_width = target_size
    scale_x, scale_y = original_width / new_width, original_height / new_height
    denormalized_points = predicted_points * np.array([scale_x, scale_y])
    
    return denormalized_points.astype(int)


if __name__ == '__main__':
    # Load the test image
    test_image_path = 'path_to_test_image.jpg'
    test_image = cv2.imread(test_image_path)
    
    # Path to the trained model
    model_path = 'path_to_trained_model.h5'
    
    # Get the predicted corners
    predicted_corners = corners(test_image, model_path)
    
    # Visualize the predicted corners on the image
    for point in predicted_corners[0]:
        cv2.circle(test_image, tuple(point), 5, (0, 255, 0), -1)
    
    # Display the image with predicted corners
    cv2.imshow('Predicted Corners', test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # If you have the ground truth points for the test image, you can calculate the MSE
    ground_truth_points = np.array([...])  # Replace with actual ground truth points
    mse = mean_squared_error(ground_truth_points, predicted_corners[0])
    print(f"Mean Squared Error: {mse}")