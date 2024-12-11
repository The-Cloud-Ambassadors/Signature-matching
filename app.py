from flask import Flask, request, jsonify
import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from skimage.metrics import structural_similarity as ssim
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@dataclass
class SignatureMatchingConfig:
    """Configuration parameters for signature matching"""
    MIN_FEATURE_SCORE: float = 30.0
    MIN_MATCHES: int = 25
    ORB_FEATURES: int = 4000
    DISTANCE_THRESHOLD: float = 40.0
    FEATURE_WEIGHT: float = 0.4
    SSIM_WEIGHT: float = 0.3
    HISTOGRAM_WEIGHT: float = 0.2
    TEMPLATE_WEIGHT: float = 0.1
    TARGET_SIZE: Tuple[int, int] = (300, 150)

class SignatureMatcher:
    def __init__(self, config: SignatureMatchingConfig):
        self.config = config
        self.orb = cv2.ORB_create(nfeatures=config.ORB_FEATURES)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the image for better feature detection"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            resized = cv2.resize(edges, self.config.TARGET_SIZE)
            return resized
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            raise

    def calculate_feature_score(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate ORB feature matching score"""
        try:
            kp1, des1 = self.orb.detectAndCompute(img1, None)
            kp2, des2 = self.orb.detectAndCompute(img2, None)

            if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
                return 0.0

            matches = self.bf.knnMatch(des1, des2, k=2)
            good_matches = [
                m for m, n in matches if m.distance < 0.75 * n.distance
            ]

            if len(good_matches) < self.config.MIN_MATCHES:
                return 0.0

            return (len(good_matches) / min(len(kp1), len(kp2))) * 100
        except Exception as e:
            logger.error(f"Error in feature matching: {str(e)}")
            return 0.0

    def calculate_histogram_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate histogram similarity score"""
        try:
            hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256]).flatten()
            hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256]).flatten()
            hist1 /= np.sum(hist1)
            hist2 /= np.sum(hist2)
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return max(correlation, 0) * 100
        except Exception as e:
            logger.error(f"Error in histogram similarity calculation: {str(e)}")
            return 0.0

    def calculate_ssim_score(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate structural similarity score"""
        try:
            score = ssim(img1, img2)
            return score * 100
        except Exception as e:
            logger.error(f"Error in SSIM calculation: {str(e)}")
            return 0.0

    def calculate_template_score(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate template matching score"""
        try:
            result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            return max_val * 100
        except Exception as e:
            logger.error(f"Error in template matching: {str(e)}")
            return 0.0

    def match_signatures(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate overall signature matching score"""
        try:
            # Preprocess images
            proc_img1 = self.preprocess_image(img1)
            proc_img2 = self.preprocess_image(img2)

            # Calculate individual scores
            feature_score = self.calculate_feature_score(proc_img1, proc_img2)
            histogram_score = self.calculate_histogram_similarity(proc_img1, proc_img2)
            ssim_score = self.calculate_ssim_score(proc_img1, proc_img2)
            template_score = self.calculate_template_score(proc_img1, proc_img2)

            if feature_score < self.config.MIN_FEATURE_SCORE:
                return 0.0

            # Calculate combined score
            combined_score = (
                self.config.FEATURE_WEIGHT * feature_score +
                self.config.SSIM_WEIGHT * ssim_score +
                self.config.HISTOGRAM_WEIGHT * histogram_score +
                self.config.TEMPLATE_WEIGHT * template_score
            )

            return min(combined_score, 100)
        except Exception as e:
            logger.error(f"Error in signature matching: {str(e)}")
            return 0.0

def load_image_from_request(file) -> Optional[np.ndarray]:
    """Load and validate image from request"""
    try:
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        return img
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        return None

@app.route('/match-signatures', methods=['POST'])
def match_signatures():
    try:
        contract_file = request.files.get('contract')
        id_file = request.files.get('id')

        if not contract_file or not id_file:
            return jsonify({"error": "Both contract and ID files are required"}), 400

        contract_image = load_image_from_request(contract_file)
        id_image = load_image_from_request(id_file)

        if contract_image is None or id_image is None:
            return jsonify({"error": "Invalid image files"}), 400

        config = SignatureMatchingConfig()
        matcher = SignatureMatcher(config)

        similarity_score = matcher.match_signatures(contract_image, id_image)

        return jsonify({
            "similarity_score": similarity_score,
            "match_confidence": "high" if similarity_score > 80 else "medium" if similarity_score > 60 else "low"
        }), 200

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
