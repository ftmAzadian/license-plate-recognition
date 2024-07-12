import numpy as np
import cv2

def mask(image: np.ndarray, points: np.ndarray, cover: np.ndarray) -> np.ndarray:
    """
    Cover the license plate area in the image with a given cover image.

    :param image: The original image with the license plate.
    :param points: The coordinates of the four corners of the license plate.
    :param cover: The cover image to obscure the license plate.
    :return: The new image with the license plate covered.
    """
    # Get the four points from the input array
    top_left, top_right, bottom_right, bottom_left = points

    # Define the order of the points for Homography estimation
    src_points = np.float32([top_left, top_right, bottom_right, bottom_left])
    dst_points = np.float32([[0, 0], [cover.shape[1], 0], [cover.shape[1], cover.shape[0]], [0, cover.shape[0]]])
    
    # Estimate the Homography matrix using findHomography
    H, _ = cv2.findHomography(dst_points,src_points, cv2.RANSAC, 5.0)
    # Warp the cover image using the Homography matrix
    new_cover_image = cv2.warpPerspective(cover, H, (image.shape[1], image.shape[0]))
    
    # Replace the license plate region in the original image with the warped cover image
    mask = cv2.polylines(np.ones_like(image), [points], True, 0)
    masked_image = cv2.bitwise_or(image, mask)

    # Add the warped cover image to the masked original image
    result = cv2.add(masked_image, new_cover_image)

    return result

# Example usage
if __name__ == "__main__":
    # Load the original image
    original_image = cv2.imread('7bd268bf-30.jpg')
    
    height, width = original_image.shape[:2]
    # Define the coordinates of the four corners of the license plate
    normalized_points = np.array([[0.39481409001956946, 0.6477495107632094], [0.598825831702544, 0.6438356164383562], 
                                  [0.5973581213307241, 0.6986301369863014], [0.3933463796477495, 0.6986301369863014]])
    
    points = (normalized_points * np.array([width, height])).astype(int)
    
    # Load the cover image
    cover_image = cv2.imread('kntu.jpg', cv2.IMREAD_UNCHANGED)  # Use IMREAD_UNCHANGED to include the alpha channel if present
    
    # Cover the license plate
    new_image = mask(original_image, points, cover_image)
    
    # Save the new image
    cv2.imwrite('new_image_with_covered_plate.jpg', new_image)
    
    # Display the new image
    cv2.imshow('New Image', new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()