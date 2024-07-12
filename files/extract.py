import numpy as np
import cv2

def extract(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Extract the license plate from an image and correct its perspective.

    :param image: The original image containing the license plate.
    :param points: The coordinates of the four corners of the license plate.
    :return: The extracted license plate image with corrected perspective and aspect ratio of 4.5.
    """
    # Get the four points from the input array
    top_left, top_right, bottom_right, bottom_left = points

    # Calculate the width of the new license plate image
    # The width is the maximum distance between the bottom-right and bottom-left points
    # or the top-right and top-left points
    width = max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left))

    # Calculate the height based on the aspect ratio of 4.5
    height = width / 4.5

    # Define the destination points for the perspective transform
    # The destination points are arranged in a rectangle with the calculated width and height
    dst_points = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)

    # Estimate the Homography matrix using findHomography
    src_points = np.float32([top_left, top_right, bottom_right, bottom_left])
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    # Warp the license plate region using the Homography matrix
    # The size of the warped image is determined by the width and height calculated above
    new_license_plate_image = cv2.warpPerspective(image, H, (int(width), int(height)))

    return new_license_plate_image



# Example usage
if __name__ == "__main__":
    # Load the original image
    original_image = cv2.imread('7bd268bf-30.jpg')
    
    height, width = original_image.shape[:2]
    # Define the coordinates of the four corners of the license plate
    normalized_points = np.array([[0.39481409001956946, 0.6477495107632094], [0.598825831702544, 0.6438356164383562], 
                                  [0.5973581213307241, 0.6986301369863014], [0.3933463796477495, 0.6986301369863014]])
    
    points = (normalized_points * np.array([width, height])).astype(int)
    
    # Extract the license plate
    license_plate_image = extract(original_image, points)
    
    # Save the extracted license plate image
    cv2.imwrite('extracted_license_plate.jpg', license_plate_image)
    
    # Display the extracted license plate image
    cv2.imshow('Extracted License Plate', license_plate_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
