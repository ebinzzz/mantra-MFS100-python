from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import cv2
import base64
import numpy as np
import requests
import json
import sqlite3
import warnings
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os

# Suppress SSL warnings for self-signed certificates
from urllib3.exceptions import InsecureRequestWarning
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

app = Flask(__name__)

app.secret_key = 'your-secret-key-here'  # Change this in production

class FingerprintManager:
    def __init__(self, base_url="https://mantra-mfs500-python-1.onrender.com/mfscan/", db_path="fingerprints.db"):
        self.base_url = base_url
        self.db_path = db_path
        self.init_database()
        
        # Quality thresholds
        self.MIN_IMAGE_QUALITY = 30
        self.MIN_CONTRAST_THRESHOLD = 20
        self.MIN_SHARPNESS_THRESHOLD = 100
        self.MAX_NOISE_LEVEL = 0.3
        
        # Accuracy testing parameters
        self.test_results = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }
    
    def check_and_update_database_schema(self):
        """Check and update database schema if needed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("PRAGMA table_info(verification_log)")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            if 'user_id' not in column_names:
                cursor.execute("DROP TABLE IF EXISTS verification_log")
                cursor.execute('''
                    CREATE TABLE verification_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        verification_time TEXT,
                        status TEXT,
                        similarity_score REAL,
                        finger_position TEXT
                    )
                ''')
                
        except sqlite3.Error as e:
            print(f"Database schema check error: {e}")
        
        conn.commit()
        conn.close()
    
    def init_database(self):
        """Initialize SQLite database with finger position validation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                finger_position TEXT NOT NULL,
                fingerprint_hash TEXT NOT NULL,
                bitmap_data TEXT NOT NULL,
                quality INTEGER,
                features TEXT,
                template_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, finger_position)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS verification_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                verification_time TEXT,
                status TEXT,
                similarity_score REAL,
                finger_position TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        self.check_and_update_database_schema()
    
    def post_mfscan_request(self, method, data=None):
        """Send POST request to MFScan API"""
        url = self.base_url + method
        headers = {'Content-Type': 'application/json'}
        
        try:
            response = requests.post(
                url,
                json=data if data else {},
                headers=headers,
                verify=False,
                timeout=30
            )
            
            return {
                'success': response.status_code == 200,
                'data': response.json() if response.content else None,
                'status_code': response.status_code
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e),
                'data': None
            }
    
    def test_connection(self):
        """Test connection to MFScan service"""
        response = self.post_mfscan_request("info")
        
        if response['success']:
            return True, response.get('data', {})
        else:
            return False, response.get('error', 'Unknown error')
    
    def capture_fingerprint(self, quality=85, timeout=15):
        """Capture fingerprint with higher quality requirement"""
        data = {
            "Quality": quality,
            "TimeOut": timeout
        }
        
        response = self.post_mfscan_request("capture", data)
        
        if response['success'] and response['data']:
            fingerprint_data = response['data']
            
            error_code = fingerprint_data.get('ErrorCode')
            error_desc = fingerprint_data.get('ErrorDescription', '')
            
            if (error_code == 0 or error_desc.lower() == 'success') and fingerprint_data.get('BitmapData'):
                return {
                    'success': True,
                    'bitmap_data': fingerprint_data.get('BitmapData'),
                    'quality': fingerprint_data.get('Quality'),
                    'width': fingerprint_data.get('Width'),
                    'height': fingerprint_data.get('Height'),
                    'error_code': error_code,
                    'error_description': error_desc
                }
            else:
                return {
                    'success': False,
                    'error': f"Capture failed: {error_desc} (Code: {error_code})",
                    'error_code': error_code
                }
        else:
            error_msg = response.get('error', 'API request failed')
            return {
                'success': False,
                'error': error_msg
            }

    def load_fingerprint_from_base64(self, base64_data: str):
        """Load fingerprint image from base64 string"""
        try:
            img_data = base64.b64decode(base64_data)
            nparr = np.frombuffer(img_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            return None

    def create_fingerprint_hash(self, bitmap_data: str) -> str:
        """Create a hash of the fingerprint data for quick comparison"""
        return hashlib.sha256(bitmap_data.encode()).hexdigest()

    def preprocess_fingerprint(self, image):
        """Enhanced preprocessing for better fingerprint comparison"""
        if image is None:
            return None
            
        image = cv2.resize(image, (256, 256))
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        return image

    def extract_ridge_orientations(self, image):
        """Extract ridge orientation patterns"""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        orientations = np.arctan2(sobely, sobelx)
        orientations = np.abs(orientations * 180 / np.pi)
        orientations[orientations > 90] = 180 - orientations[orientations > 90]
        
        return orientations

    def ridge_orientation_similarity(self, img1, img2):
        """Compare ridge orientation patterns"""
        orient1 = self.extract_ridge_orientations(img1)
        orient2 = self.extract_ridge_orientations(img2)
        
        hist1, _ = np.histogram(orient1.flatten(), bins=18, range=(0, 90))
        hist2, _ = np.histogram(orient2.flatten(), bins=18, range=(0, 90))
        
        hist1 = hist1.astype(float) / np.sum(hist1)
        hist2 = hist2.astype(float) / np.sum(hist2)
        
        correlation = np.corrcoef(hist1, hist2)[0, 1]
        
        block_size = 32
        spatial_correlations = []
        
        for i in range(0, orient1.shape[0] - block_size, block_size):
            for j in range(0, orient1.shape[1] - block_size, block_size):
                block1 = orient1[i:i+block_size, j:j+block_size]
                block2 = orient2[i:i+block_size, j:j+block_size]
                
                if np.std(block1) > 5 and np.std(block2) > 5:
                    block_corr = np.corrcoef(block1.flatten(), block2.flatten())[0, 1]
                    if not np.isnan(block_corr):
                        spatial_correlations.append(block_corr)
        
        spatial_score = np.mean(spatial_correlations) if spatial_correlations else 0
        final_score = 0.4 * correlation + 0.6 * spatial_score
        
        return max(0, final_score) if not np.isnan(final_score) else 0

    def extract_robust_keypoints(self, image):
        """Extract keypoints using ORB detector"""
        orb = cv2.ORB_create(nfeatures=300, scaleFactor=1.2, nlevels=8)
        kp_orb, desc_orb = orb.detectAndCompute(image, None)
        
        return kp_orb, desc_orb

    def advanced_feature_matching(self, img1, img2):
        """Advanced feature matching with geometric verification"""
        kp1, desc1 = self.extract_robust_keypoints(img1)
        kp2, desc2 = self.extract_robust_keypoints(img2)
        
        if desc1 is None or desc2 is None or len(desc1) < 10 or len(desc2) < 10:
            return 0.0
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        good_matches = [m for m in matches if m.distance < 50]
        
        if len(good_matches) < 4:
            return 0.0
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        try:
            _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            inliers = np.sum(mask)
            
            inlier_ratio = inliers / len(good_matches)
            match_density = len(good_matches) / min(len(kp1), len(kp2))
            
            score = 0.7 * inlier_ratio + 0.3 * min(1.0, match_density * 10)
            return score
            
        except:
            match_ratio = len(good_matches) / min(len(kp1), len(kp2))
            return min(1.0, match_ratio * 5)

    def local_binary_pattern_similarity(self, img1, img2):
        """Compare Local Binary Patterns"""
        def calculate_lbp(image):
            rows, cols = image.shape
            lbp = np.zeros((rows-2, cols-2), dtype=np.uint8)
            
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    center = image[i, j]
                    binary = 0
                    
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j+1], image[i+1, j+1], image[i+1, j],
                        image[i+1, j-1], image[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            binary |= (1 << k)
                    
                    lbp[i-1, j-1] = binary
            
            return lbp
        
        lbp1 = calculate_lbp(img1)
        lbp2 = calculate_lbp(img2)
        
        hist1 = cv2.calcHist([lbp1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([lbp2], [0], None, [256], [0, 256])
        
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0, correlation)

    def match_fingerprints(self, img1, img2, threshold=0.55):
        """Enhanced fingerprint matching"""
        processed1 = self.preprocess_fingerprint(img1)
        processed2 = self.preprocess_fingerprint(img2)
        
        if processed1 is None or processed2 is None:
            return False, 0.0, {}

        scores = {}
        
        try:
            scores['ridge_orientation'] = self.ridge_orientation_similarity(processed1, processed2)
        except Exception:
            scores['ridge_orientation'] = 0.0
        
        try:
            scores['advanced_features'] = self.advanced_feature_matching(processed1, processed2)
        except Exception:
            scores['advanced_features'] = 0.0
        
        try:
            scores['lbp'] = self.local_binary_pattern_similarity(processed1, processed2)
        except Exception:
            scores['lbp'] = 0.0
        
        try:
            img1_norm = cv2.normalize(processed1, None, 0, 255, cv2.NORM_MINMAX)
            img2_norm = cv2.normalize(processed2, None, 0, 255, cv2.NORM_MINMAX)
            template_score = cv2.matchTemplate(img1_norm, img2_norm, cv2.TM_CCOEFF_NORMED)[0][0]
            scores['template'] = max(0, template_score)
        except Exception:
            scores['template'] = 0.0

        weights = {
            'ridge_orientation': 0.45,
            'advanced_features': 0.30,
            'lbp': 0.15,
            'template': 0.10
        }
        
        combined_score = sum(weights[method] * scores[method] for method in scores.keys())
        
        critical_failures = 0
        
        if scores['ridge_orientation'] < 0.2:
            critical_failures += 1
        
        if scores['advanced_features'] < 0.1:
            critical_failures += 1
        
        if critical_failures >= 2:
            combined_score *= 0.5
        elif critical_failures == 1:
            combined_score *= 0.8

        is_match = combined_score >= threshold
        return is_match, combined_score, scores

    def extract_minutiae_features(self, image):
        """Extract minutiae features using ORB detector"""
        if image is None:
            return [], None
            
        orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
        keypoints, descriptors = orb.detectAndCompute(image, None)
        
        minutiae = []
        for kp in keypoints:
            minutiae.append({
                'x': int(kp.pt[0]),
                'y': int(kp.pt[1]),
                'angle': kp.angle,
                'response': kp.response
            })
        
        return minutiae, descriptors

    def enroll_user(self, user_id: str, finger_position: str = "unknown"):
        """Enroll a new user fingerprint"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM fingerprints WHERE user_id = ? AND finger_position = ?", 
                      (user_id, finger_position))
        if cursor.fetchone():
            conn.close()
            return False, f"User '{user_id}' with finger position '{finger_position}' already exists!"
        conn.close()
        
        capture_result = self.capture_fingerprint()
        if not capture_result['success']:
            return False, f"Capture failed: {capture_result.get('error', 'Unknown error')}"
        
        bitmap_data = capture_result['bitmap_data']
        quality = capture_result.get('quality', 0)
        
        img = self.load_fingerprint_from_base64(bitmap_data)
        if img is None:
            return False, "Failed to process captured fingerprint"
        
        minutiae, descriptors = self.extract_minutiae_features(img)
        
        if len(minutiae) < 10:
            return False, f"Insufficient minutiae detected: {len(minutiae)} (minimum: 10)"
        
        fp_hash = self.create_fingerprint_hash(bitmap_data)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO fingerprints 
                (user_id, finger_position, fingerprint_hash, bitmap_data, quality, features, template_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, 
                finger_position, 
                fp_hash, 
                bitmap_data, 
                quality,
                json.dumps(minutiae),
                json.dumps({'descriptors_shape': descriptors.shape if descriptors is not None else None})
            ))
            
            conn.commit()
            return True, f"User '{user_id}' enrolled successfully! Quality: {quality}, Minutiae: {len(minutiae)}"
            
        except sqlite3.Error as e:
            return False, f"Database error: {e}"
        finally:
            conn.close()

    def verify_user(self):
        """Verify a user by capturing their fingerprint"""
        capture_result = self.capture_fingerprint()
        if not capture_result['success']:
            return False, None, 0.0, f"Capture failed: {capture_result.get('error', 'Unknown error')}"
        
        bitmap_data = capture_result['bitmap_data']
        captured_img = self.load_fingerprint_from_base64(bitmap_data)
        
        if captured_img is None:
            return False, None, 0.0, "Failed to process captured fingerprint"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, finger_position, bitmap_data FROM fingerprints")
        enrolled_prints = cursor.fetchall()
        conn.close()
        
        if not enrolled_prints:
            return False, None, 0.0, "No enrolled fingerprints found!"
        
        best_match = None
        best_score = 0.0
        best_finger = None
        
        for user_id, finger_pos, stored_data in enrolled_prints:
            stored_img = self.load_fingerprint_from_base64(stored_data)
            if stored_img is None:
                continue
            
            is_match, similarity_score, _ = self.match_fingerprints(captured_img, stored_img)
            
            if similarity_score > best_score:
                best_score = similarity_score
                best_match = user_id if is_match else None
                best_finger = finger_pos if is_match else None
        
        status = "SUCCESS" if best_match else "FAILED"
        self.log_verification(best_match or "Unknown", status, best_score, best_finger)
        
        if best_match:
            return True, best_match, best_score, f"Verified: {best_match} ({best_finger}) - Score: {best_score:.4f}"
        else:
            return False, None, best_score, f"Verification failed (Best: {best_score:.4f})"

    def log_verification(self, user_id: str, status: str, score: float, finger_position: str = None):
        """Log verification attempt to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO verification_log 
                (user_id, verification_time, status, similarity_score, finger_position)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, datetime.now().isoformat(), status, score, finger_position))
            
            conn.commit()
        except sqlite3.Error:
            pass
        finally:
            conn.close()

    def get_all_users(self):
        """Get all enrolled users"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_id, finger_position, quality, created_at 
            FROM fingerprints 
            ORDER BY user_id, finger_position
        ''')
        fingerprints = cursor.fetchall()
        conn.close()
        
        return fingerprints

    def delete_user(self, user_id: str, finger_position: str = None):
        """Delete a user's fingerprint(s)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if finger_position:
            cursor.execute("SELECT user_id FROM fingerprints WHERE user_id = ? AND finger_position = ?", 
                          (user_id, finger_position))
            if not cursor.fetchone():
                conn.close()
                return False, f"Fingerprint for '{user_id}' ({finger_position}) not found!"
            
            cursor.execute("DELETE FROM fingerprints WHERE user_id = ? AND finger_position = ?", 
                          (user_id, finger_position))
            message = f"Fingerprint for '{user_id}' ({finger_position}) deleted!"
        else:
            cursor.execute("SELECT COUNT(*) FROM fingerprints WHERE user_id = ?", (user_id,))
            count = cursor.fetchone()[0]
            
            if count == 0:
                conn.close()
                return False, f"No fingerprints found for user '{user_id}'!"
            
            cursor.execute("DELETE FROM fingerprints WHERE user_id = ?", (user_id,))
            message = f"All {count} fingerprints for '{user_id}' deleted!"
        
        conn.commit()
        conn.close()
        return True, message

    def get_statistics(self):
        """Get system statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM fingerprints")
        fp_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM fingerprints")
        user_count = cursor.fetchone()[0]
        
        success_count = 0
        failed_count = 0
        recent_verifications = []
        
        try:
            cursor.execute("SELECT COUNT(*) FROM verification_log WHERE status = 'SUCCESS'")
            success_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM verification_log WHERE status = 'FAILED'")
            failed_count = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT user_id, verification_time, status, similarity_score, finger_position
                FROM verification_log 
                ORDER BY verification_time DESC 
                LIMIT 10
            ''')
            recent_verifications = cursor.fetchall()
        except sqlite3.Error:
            pass
        
        conn.close()
        
        success_rate = (success_count/(success_count+failed_count)*100 if success_count+failed_count > 0 else 0)
        
        return {
            'user_count': user_count,
            'fp_count': fp_count,
            'success_count': success_count,
            'failed_count': failed_count,
            'success_rate': success_rate,
            'recent_verifications': recent_verifications
        }

# Initialize the fingerprint manager
manager = FingerprintManager()

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test_connection')
def test_connection():
    success, result = manager.test_connection()
    return jsonify({'success': success, 'result': result})

@app.route('/enroll', methods=['GET', 'POST'])
def enroll():
    if request.method == 'POST':
        user_id = request.form['user_id'].strip()
        finger_position = request.form['finger_position']
        
        if not user_id:
            flash('Please enter a valid user ID', 'error')
            return redirect(url_for('enroll'))
        
        success, message = manager.enroll_user(user_id, finger_position)
        
        if success:
            flash(message, 'success')
        else:
            flash(message, 'error')
        
        return redirect(url_for('enroll'))
    
    return render_template('enroll.html')

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        success, user_id, score, message = manager.verify_user()
        
        if success:
            flash(f"Welcome, {user_id}! Access granted. (Score: {score:.4f})", 'success')
        else:
            flash(f"Access denied. {message}", 'error')
        
        return redirect(url_for('verify'))
    
    return render_template('verify.html')

@app.route('/users')
def users():
    fingerprints = manager.get_all_users()
    return render_template('users.html', fingerprints=fingerprints)

@app.route('/delete_user', methods=['POST'])
def delete_user():
    user_id = request.form['user_id']
    finger_position = request.form.get('finger_position')
    
    success, message = manager.delete_user(user_id, finger_position if finger_position else None)
    
    if success:
        flash(message, 'success')
    else:
        flash(message, 'error')
    
    return redirect(url_for('users'))

@app.route('/statistics')
def statistics():
    stats = manager.get_statistics()
    return render_template('statistics.html', stats=stats)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    if not os.path.exists('static'):
        os.makedirs('static')
    
    app.run(debug=True, host='0.0.0.0', port=5000)