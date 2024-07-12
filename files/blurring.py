import numpy as np
import cv2

from extract import extract
from masking import mask

def blur(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Blur the license plate in an image.

    :param image: The original image containing the license plate.
    :param points: The coordinates of the four corners of the license plate.
    :return: The image with the license plate blurred.
    """
    # Extract the license plate with corrected perspective
    license_plate = extract(image, points)

    # Blur the license plate (e.g., using Gaussian blur)
    blurred_license_plate = cv2.GaussianBlur(license_plate, (25, 25), -10)

    # Overlay the blurred license plate back onto the original image
    ##new_image = mask(image, points, blurred_license_plate)
    # Get the four points from the input array
    top_left, top_right, bottom_right, bottom_left = points

    # Define the order of the points for Homography estimation
    src_points = np.float32([top_left, top_right, bottom_right, bottom_left])
    dst_points = np.float32([[0, 0], [blurred_license_plate.shape[1], 0], [blurred_license_plate.shape[1], blurred_license_plate.shape[0]], [0, blurred_license_plate.shape[0]]])
    
    # Estimate the Homography matrix using findHomography
    H, _ = cv2.findHomography(dst_points,src_points, cv2.RANSAC, 5.0)
    # Warp the cover image using the Homography matrix
    new_cover_image = cv2.warpPerspective(blurred_license_plate, H, (image.shape[1], image.shape[0]))
    cv2.imshow('Blurred License Plate', new_cover_image)
    cv2.waitKey(0)
    # Replace the license plate region in the original image with the warped cover image
    mask = cv2.polylines(np.ones_like(image), [points], True, 0)
    cv2.fillPoly(mask, [np.array(points)], (255, 255, 255))  # Fill the polygon defined by 'points' with white

    # Create an inverse mask to preserve the original content of the image
    mask_inv = cv2.bitwise_not(mask)

    # Use the inverse mask to preserve the rest of the original image
    rest_of_image = cv2.bitwise_and(image, mask_inv)
    new_image = cv2.add(rest_of_image, new_cover_image)

    return new_image

# Example usage
if __name__ == "__main__":
    # Load the original image
    original_image = cv2.imread('7bd268bf-30.jpg')
    
    height, width = original_image.shape[:2]
    # Define the coordinates of the four corners of the license plate
    normalized_points = np.array([[0.39481409001956946, 0.6477495107632094], [0.598825831702544, 0.6438356164383562], 
                                  [0.5973581213307241, 0.6986301369863014], [0.3933463796477495, 0.6986301369863014]])
    
    points = (normalized_points * np.array([width, height])).astype(int)
    
    # Blur the license plate
    blurred_image = blur(original_image, points)
    
    # Save the blurred image
    cv2.imwrite('blurred_license_plate.jpg', blurred_image)
    
    # Display the blurred image
    cv2.imshow('Blurred License Plate', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
