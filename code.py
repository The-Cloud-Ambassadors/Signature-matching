import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

class SignatureMatchingConfig:
    """Configuration parameters for signature matching"""
    MIN_FEATURE_SCORE: float = 10.0  # Reduced to allow more matches
    MIN_MATCHES: int = 10  # Reduced to increase flexibility
    ORB_FEATURES: int = 5000  # Increased to extract more features
    DISTANCE_THRESHOLD: float = 60.0  # Increased for leniency in matching
    FEATURE_WEIGHT: float = 0.7 # Increased weight for feature matching
    SSIM_WEIGHT: float = 0.5
    HISTOGRAM_WEIGHT: float = 0.45
    TARGET_SIZE: tuple = (300, 150)

class SignatureMatcher:
    def __init__(self, config: SignatureMatchingConfig):
        self.config = config
        self.orb = cv2.ORB_create(nfeatures=config.ORB_FEATURES)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def preprocess_signature(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the extracted signature region"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 120)  # Adjusted thresholds for better edge detection
        resized = cv2.resize(edges, self.config.TARGET_SIZE)
        return resized

    def calculate_feature_score(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate ORB feature matching score"""
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
            return 0.0

        matches = self.bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.85 * n.distance]  # Adjusted ratio for leniency

        if len(good_matches) < self.config.MIN_MATCHES:
            return 0.0

        return (len(good_matches) / min(len(kp1), len(kp2))) * 100

    def calculate_ssim_score(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate structural similarity score"""
        score = ssim(img1, img2)
        return score * 100

    def calculate_histogram_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate histogram similarity score"""
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256]).flatten()
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256]).flatten()
        hist1 /= np.sum(hist1)
        hist2 /= np.sum(hist2)
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(correlation, 0) * 100

    def match_signatures(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate overall signature matching score"""
        feature_score = self.calculate_feature_score(img1, img2)
        ssim_score = self.calculate_ssim_score(img1, img2)
        histogram_score = self.calculate_histogram_similarity(img1, img2)

        if feature_score < self.config.MIN_FEATURE_SCORE:
            return 0.0

        combined_score = (
            self.config.FEATURE_WEIGHT * feature_score +
            self.config.SSIM_WEIGHT * ssim_score +
            self.config.HISTOGRAM_WEIGHT * histogram_score
        )

        return min(combined_score, 100)

def extract_signature_region(image: np.ndarray) -> np.ndarray:
    """Extract the region of interest (signature) from the image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours and sort by area
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 100 and h > 50:  # Adjust heuristics for signature dimensions
            cropped = image[y:y+h, x:x+w]
            return cropped

    raise ValueError("No suitable signature region found")

def load_image_from_file(file_path: str) -> np.ndarray:
    """Load an image from a file"""
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError("Invalid image file")
    return image

# Example usage
if __name__ == "__main__":
    # Paths to your images
    contract_image_path = "/content/sample_signature..jpg"  # Replace with your contract image path
    id_image_path = "/content/Screenshot 2024-11-27 110641.png"  # Replace with your ID image path

    # Load images
    contract_image = load_image_from_file(contract_image_path)
    id_image = load_image_from_file(id_image_path)

    # Extract signature regions
    contract_signature = extract_signature_region(contract_image)
    id_signature = extract_signature_region(id_image)

    # Initialize matcher
    config = SignatureMatchingConfig()
    matcher = SignatureMatcher(config)

    # Preprocess signatures
    contract_preprocessed = matcher.preprocess_signature(contract_signature)
    id_preprocessed = matcher.preprocess_signature(id_signature)

    # Match signatures
    similarity_score = matcher.match_signatures(contract_preprocessed, id_preprocessed)

    print(f"Similarity Score: {similarity_score}")
