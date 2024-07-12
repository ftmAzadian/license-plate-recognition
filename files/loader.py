import numpy as np
import cv2
from typing import Tuple

def augment(image: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Apply random transformations
    rows, cols, _ = image.shape
    
    # Randomly choose the type of transformation to apply
    transform_type = np.random.choice(['shift', 'blur', 'crop', 'resize', 'rotate', 'contrast', 'perspective'])

    if transform_type == 'shift':
        # Random shift (translation)
        dx, dy = np.random.randint(-5, 5, 2)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        image = cv2.warpAffine(image, M, (cols, rows))
        points += np.float32([dx, dy])
    elif transform_type == 'blur':
        # Random blur
        ksize = np.random.choice([3, 5, 7])
        if ksize % 2 == 1:  # Ensure kernel size is odd
            image = cv2.blur(image, (ksize, ksize))
    elif transform_type == 'crop':
        # Random crop
        x, y, w, h = np.random.randint(0, 20), np.random.randint(0, 20), np.random.randint(80, 100), np.random.randint(80, 100)
        image = image[y:y+h, x:x+w]
        points -= np.float32([x, y])
    elif transform_type == 'resize':
        # Random resize
        scale = np.random.uniform(0.9, 1.1)
        image = cv2.resize(image, (int(cols * scale), int(rows * scale)))
        points *= scale
    elif transform_type == 'rotate':
        # Random rotation
        angle = np.random.randint(-5, 5)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        image = cv2.warpAffine(image, M, (cols, rows))
        # Update points using the rotation matrix
        points = np.dot(M, np.vstack((points.T, np.ones(points.shape[0])))).T[:, :2]
    elif transform_type == 'contrast':
        # Random contrast adjustment
        alpha = np.random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=alpha)
    elif transform_type == 'perspective':
        # Random perspective transformation
        pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
        pts2 = pts1 + np.random.randint(-5, 5, (4, 2))
        M = cv2.getPerspectiveTransform(pts1, pts2)
        image = cv2.warpPerspective(image, M, (cols, rows))
        # Update points using the perspective transformation matrix
        points = cv2.perspectiveTransform(points.reshape(1, -1, 2), M).reshape(-1, 2)

    # Ensure points are within image boundaries
    points = points.astype(int)
    points = np.clip(points, [0, 0], [cols - 1, rows - 1])

    return image, points

def resize(image: np.ndarray, points: np.ndarray, target_size=(224, 224)) -> Tuple[np.ndarray, np.ndarray]:
    # Resize image to target size while maintaining aspect ratio
    rows, cols, _ = image.shape
    new_rows, new_cols = target_size

    # Calculate the aspect ratio of the original image and the target image
    original_aspect_ratio = cols / rows
    target_aspect_ratio = new_cols / new_rows

    # Determine the dimensions for resizing
    if original_aspect_ratio > target_aspect_ratio:
        # Wide image, height is the limiting factor
        new_height = new_rows
        new_width = int(new_rows * original_aspect_ratio)
    else:
        # Tall or square image, width is the limiting factor
        new_width = new_cols
        new_height = int(new_cols / original_aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Calculate padding
    pad_top = int((new_rows - new_height) / 2)
    pad_bottom = new_rows - new_height - pad_top
    pad_left = int((new_cols - new_width) / 2)
    pad_right = new_cols - new_width - pad_left

    # Pad the image to the target size
    padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)

    # Adjust the points accordingly
    points[:, 0] = points[:, 0] * (new_width / cols) + pad_left
    points[:, 1] = points[:, 1] * (new_height / rows) + pad_top

    return padded_image, points

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def load(dir_name: str, image_size=(224, 224), validation_split=0.1, test_split=0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    images = []  # List to hold images
    points_list = []  # List to hold points
    
    # Load all images and points from the given directory
    for filename in os.listdir(dir_name):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Assuming image files are JPEG or PNG
            image_path = os.path.join(dir_name, filename)
            image = cv2.imread(image_path)
            # Assuming the points are stored in a text file with the same name as the image file
            points_path = os.path.join(dir_name, os.path.splitext(filename)[0] + '.txt')
            points = np.loadtxt(points_path).reshape(-1, 2)  # Example points loading
            
            # Resize and normalize the image
            image = cv2.resize(image, image_size)
            image = image / 255.0  # Normalize pixel values to [0, 1]
            
            images.append(image)
            points_list.append(points)
    
    # Convert lists to numpy arrays
    X = np.array(images)
    y = np.array(points_list)
    
    # Perform train-test-validation split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split/(1-test_split), random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Optional
class PlateDataGenerator:
    def __init__(self, dir_name, batch_size=32, target_size=(224, 224)):
        self.dir_name = dir_name
        self.batch_size = batch_size
        self.target_size = target_size
        # Load all data
        self.X, self.y = load(dir_name)
        
    def __len__(self):
        return len(self.X) // self.batch_size
    
    def __getitem__(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        batch_X = self.X[start:end]
        batch_y = self.y[start:end]
        # Apply augmentation
        augmented_X, augmented_y = [], []
        for image, points in zip(batch_X, batch_y):
            aug_image, aug_points = augment(image, points)
            augmented_X.append(aug_image)
            augmented_y.append(aug_points)
        # Resize images
        resized_X, resized_y = [], []
        for image, points in zip(augmented_X, augmented_y):
            resized_image, resized_points = resize(image, points, self.target_size)
            resized_X.append(resized_image)
            resized_y.append(resized_points)
        
        return np.array(resized_X), np.array(resized_y)