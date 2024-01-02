import cv2
import matplotlib.pyplot as plt

# Load the image
image_path = 'images/lena.jpg'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found.")
else:
    # Apply Sobel operator
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel Edge Detection on the Y axis

    # Convert to absolute values
    abs_sobelx = cv2.convertScaleAbs(sobelx)
    abs_sobely = cv2.convertScaleAbs(sobely)

    # Combine the two images
    combined_sobel = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)

    # Display the images
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
    plt.subplot(2, 2, 2), plt.imshow(abs_sobelx, cmap='gray'), plt.title('Sobel X')
    plt.subplot(2, 2, 3), plt.imshow(abs_sobely, cmap='gray'), plt.title('Sobel Y')
    plt.subplot(2, 2, 4), plt.imshow(combined_sobel, cmap='gray'), plt.title('Combined Sobel')
    plt.show()
