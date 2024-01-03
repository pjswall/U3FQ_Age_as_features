import cv2
import numpy as np

def analyze_histogram(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate histogram
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # You might define 'quality' based on certain criteria, like how spread out the histogram is (contrast)
    # or whether it's too far to the low end (underexposed) or high end (overexposed)

def multi_scale_retinex(img, sigma_list):
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += cv2.GaussianBlur(np.log10(img + 1), (0, 0), sigma)

    retinex = retinex / len(sigma_list)

    return cv2.normalize(np.exp(retinex) - 1, None, 0, 255, cv2.NORM_MINMAX)

def main():
    image = cv2.imread('./_cnsifd_s_922.jpg')

    analyze_histogram(image)

    # Multi-Scale Retinex
    sigma_list = [15, 80, 250]  # These values can be tuned
    retinex_image = multi_scale_retinex(image, sigma_list)

    # Compare the original and processed images
    cv2.imshow('Original Image', image)
    cv2.imshow('Retinex Image', retinex_image)
    cv2.waitKey(0)  # Wait for a keyboard event
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
