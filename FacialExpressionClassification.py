import cv2 as cv
import sys
import joblib
import numpy as np
import time # 导入time模块用于测量延迟
from skimage.feature import local_binary_pattern, hog
from tensorflow.keras.models import load_model
from FaceDetector.DNN_FaceDetector import DnnDetector

# ----------------------------------------------------------------------
# 1. 全局变量/配置 (移除单模型选择的 Trackbar 相关变量)
# ----------------------------------------------------------------------
CAM_INDEX = 0  # Change this if you have multiple cameras

windows_name = "Facial Emotion Recognition - Triple Model Output" # 更改窗口名称

# TB_Model, TB_Model_max_value, TB_Model_default, model = [REMOVED]

FACE_DETECTOR_ROOT = "FaceDetector"
dnn_detector = DnnDetector(root = FACE_DETECTOR_ROOT)

# 情绪类别
CLASSES = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# 全局模型变量
knn_model = None
svm_model = None
lbp_scaler = None
hog_scaler = None
mini_x_model = None


def load_models():
    """加载所有模型和 Scalers"""
    global knn_model, svm_model, lbp_scaler, hog_scaler, mini_x_model

    knn_model_path = "Model/knn_lbp_model.joblib"
    svm_model_path = "Model/svm_hog_model.joblib"
    lbp_scaler_path = "Model/scaler_lbp.joblib"
    hog_scaler_path = "Model/scaler_hog.joblib"
    mini_x_path = "Model/best_model.keras"

    try:
        # NOTE: KNN model is loaded. For confidence, we'll try predict_proba.
        knn_model = joblib.load(knn_model_path)
        print("[INFO] KNN model loaded:", knn_model_path)
    except Exception as e:
        print(f"[WARN] Fail to load KNN model: {e}")

    try:
        # NOTE: SVM model is loaded. For confidence, we'll try predict_proba, which requires probability=True in training.
        svm_model = joblib.load(svm_model_path)
        print("[INFO] SVM model loaded:", svm_model_path)
    except Exception as e:
        print(f"[WARN] Fail to load SVM model: {e}")

    try:
        lbp_scaler = joblib.load(lbp_scaler_path)
        print("[INFO] LBP Scaler loaded:", lbp_scaler_path)
    except Exception as e:
        print(f"[WARN] Fail to load LBP scaler: {e}")

    try:
        hog_scaler = joblib.load(hog_scaler_path)
        print("[INFO] HOG Scaler loaded:", hog_scaler_path)
    except Exception as e:
        print(f"[WARN] Fail to load HOG scaler: {e}")

    try:
        mini_x_model = load_model(mini_x_path)
        print("[INFO] Mini-Xception model loaded:", mini_x_path)
    except Exception as e:
        print(f"[WARN] Fail to load Mini-Xception model: {e}")


# --- (initialize_stream, grab_preprocess, load_images, CreateGUI, LBP_Features_extract, HOG_Features_extract remain UNCHANGED or REMOVED) ---

def initialize_stream():
    global cap
    global frame_in
    global isGrayCamera

    # Open camera (change CAM_INDEX if needed)
    cap = cv.VideoCapture(CAM_INDEX)

    if cap.isOpened() == False:
        sys.exit("Error opening video stream")

    # Grab one frame to inspect shape
    status, frame_in = cap.read()
    if status != True:
        sys.exit("Error reading video stream")

    # If frame has only 1 channel, treat as gray camera
    if len(frame_in.shape) == 2 or frame_in.shape[2] == 1:
        isGrayCamera = True
    else:
        isGrayCamera = False


def grab_preprocess():
    global frame_in, frame_gray

    status, frame_in = cap.read()
    if not status:
        return False

    if isGrayCamera:
        # Camera already gives grayscale
        if len(frame_in.shape) == 3 and frame_in.shape[2] == 3:
            frame_gray = cv.cvtColor(frame_in, cv.COLOR_BGR2GRAY)
        else:
            frame_gray = frame_in
    else:
        # Color camera - convert to grayscale for feature detection
        frame_gray = cv.cvtColor(frame_in, cv.COLOR_BGR2GRAY)
    return True

# NOTE: Since we are running triple prediction in one function, 
# the individual 'predict_emotion_knn', 'predict_emotion_svm', 'predict_emotion_minix' are not needed, 
# but we'll keep them for the classify_image function if needed.
# For the main loop, we'll use a new function that returns confidence and latency.


def LBP_Features_extract(image):
    height,width = (48,48)
    FEATURES = []
    P = 8
    R = 1
    LBP_Method = 'uniform'
    block_size = (8,8)
    block_height = height // block_size[0]
    block_width = width // block_size[1]
    n_bins = P + 2
    for i in range(block_size[0]):
        for j in range(block_size[1]):
            start_i = i * block_height
            end_i = start_i +block_height
            start_j = j * block_width
            end_j = start_j + block_width
            block = image[start_i:end_i, start_j:end_j]

            lbp = local_binary_pattern(block,P,R,LBP_Method)
            hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
            FEATURES.extend(hist)
    
    FEATURES = np.array(FEATURES,dtype = np.float32)
    return FEATURES

def HOG_Features_extract(image):
    HOG_FEATURES = []
    Orientation = 9
    Pixels = (8, 8)
    Cells = (2, 2)
    HOG_FEATURES = hog(image,
                       orientations=Orientation,
                       pixels_per_cell=Pixels,
                       cells_per_block=Cells,
                       block_norm='L2-Hys',
                       visualize=False,
                       channel_axis=None)
    HOG_FEATURES = np.array(HOG_FEATURES,dtype = np.float32)
    return HOG_FEATURES

# ----------------------------------------------------------------------
# 2. 新预测函数: 包含置信度和延迟
# ----------------------------------------------------------------------
def classify_face_all_models_with_metrics(face_resized):
    results = {}

    # --- KNN Prediction ---
    start_time_knn = time.time()
    if (knn_model is not None) and (lbp_scaler is not None):
        lbp_features = LBP_Features_extract(face_resized)
        feature = lbp_features.reshape(1,-1).astype("float32")
        feature_scaled = lbp_scaler.transform(feature)
        
        try:
            probs = knn_model.predict_proba(feature_scaled)[0]
            cls_id = int(np.argmax(probs))
            confidence = probs[cls_id]
            label = CLASSES.get(cls_id, str(cls_id))
        except AttributeError:
            # Fallback if predict_proba is not available
            y_pred = knn_model.predict(feature_scaled)[0]
            label = CLASSES.get(int(y_pred), str(y_pred))
            confidence = 1.0 # Placeholder for unavailable confidence
        except Exception:
            label = "KNN Err"
            confidence = 0.0
    else:
        label = "KNN N/A"
        confidence = 0.0
    end_time_knn = time.time()
    latency_knn = (end_time_knn - start_time_knn) * 1000 # ms
    results["KNN"] = {"label": label, "conf": confidence, "ms": latency_knn}

    # --- SVM Prediction ---
    start_time_svm = time.time()
    if (svm_model is not None) and (hog_scaler is not None):
        hog_features = HOG_Features_extract(face_resized)
        feature = hog_features.reshape(1, -1).astype("float32")
        feature_scaled = hog_scaler.transform(feature)
        
        try:
            # NOTE: Assumes SVM model was trained with probability=True
            probs = svm_model.predict_proba(feature_scaled)[0] 
            cls_id = int(np.argmax(probs))
            confidence = probs[cls_id]
            label = CLASSES.get(cls_id, str(cls_id))
        except AttributeError:
            # Fallback
            y_pred = svm_model.predict(feature_scaled)[0]
            label = CLASSES.get(int(y_pred), str(y_pred))
            confidence = 1.0 # Placeholder
        except Exception:
            label = "SVM Err"
            confidence = 0.0
    else:
        label = "SVM N/A"
        confidence = 0.0
    end_time_svm = time.time()
    latency_svm = (end_time_svm - start_time_svm) * 1000 # ms
    results["SVM"] = {"label": label, "conf": confidence, "ms": latency_svm}

    # --- Mini-Xception Prediction ---
    start_time_minix = time.time()
    if mini_x_model is not None:
        try:
            face_norm = face_resized.astype("float32") / 255.0
            face_input = face_norm.reshape(1, 48, 48, 1)
            probs = mini_x_model.predict(face_input, verbose=0)[0]
            cls_id = int(np.argmax(probs))
            confidence = probs[cls_id]
            label = CLASSES.get(cls_id, str(cls_id))
        except Exception:
            label = "MiniX Err"
            confidence = 0.0
    else:
        label = "MiniX N/A"
        confidence = 0.0
    end_time_minix = time.time()
    latency_minix = (end_time_minix - start_time_minix) * 1000 # ms
    results["MiniX"] = {"label": label, "conf": confidence, "ms": latency_minix}

    return results

# ----------------------------------------------------------------------
# 3. GUI 和显示修改 (移除 Trackbar, 修改 process_display)
# ----------------------------------------------------------------------

def CreateGUI():
    """仅创建窗口，移除 Trackbar"""
    cv.namedWindow(windows_name, cv.WINDOW_NORMAL)
    cv.setWindowTitle(windows_name, windows_name) # 确保标题正确

# NOTE: process_display_callback and its related code is REMOVED/DELETED

def process_display():
    """检测人脸，运行所有模型并显示结果"""
    global frame_in, frame_gray
    
    # Display the frame
    display_frame = frame_in.copy()
    gray = frame_gray
    
    # --- Face Detection ---
    faces = dnn_detector.detect_faces(display_frame)

    for (x, y, w, h) in faces:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(display_frame.shape[1], x + w), min(display_frame.shape[0], y + h)

        face_roi_gray = gray[y1:y2, x1:x2]
        if face_roi_gray.size == 0:
            continue

        # --- Pipeline: Crop/Align -> Resize (48x48) -> Run all 3 ---
        face_resized = cv.resize(face_roi_gray, (48, 48))
        
        # --- Run all 3 models ---
        results = classify_face_all_models_with_metrics(face_resized)
        
        # --- Drawing the results (Box and Text) ---
        cv.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Draw face box

        text_lines = [
            f"KNN: {results['KNN']['label']} ({results['KNN']['conf']:.2f}, {results['KNN']['ms']:.1f}ms)",
            f"SVM: {results['SVM']['label']} ({results['SVM']['conf']:.2f}, {results['SVM']['ms']:.1f}ms)",
            f"MiniX: {results['MiniX']['label']} ({results['MiniX']['conf']:.2f}, {results['MiniX']['ms']:.1f}ms)",
        ]

        # Determine where to put the text box (above or below the face)
        line_height = 18
        box_height = len(text_lines) * line_height + 5
        start_y = y1 - box_height
        
        # Adjust position if box goes out of frame at the top
        if start_y < 0:
            start_y = y2 + 5 

        # Draw a background box for the text
        box_x2 = x1 + max(len(line) for line in text_lines) * 10 + 10 # Estimate box width
        cv.rectangle(display_frame, (x1, start_y), (box_x2, start_y + box_height), (50, 50, 50), -1)

        # Write text lines
        for i, line in enumerate(text_lines):
            ty = start_y + (i * line_height) + line_height - 5
            cv.putText(display_frame, line, (x1 + 5, ty),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
            
    cv.imshow(windows_name, display_frame)


# ----------------------------------------------------------------------
# 4. 主循环修改
# ----------------------------------------------------------------------
def video_classification():
    initialize_stream()
    CreateGUI() # 仅创建窗口

    while True:
        if not grab_preprocess():
            break

        process_display() # 调用新的显示函数

        if cv.waitKey(5) >=0:
            break

    cap.release()
    cv.destroyAllWindows()


# --- (classify_image function remains UNCHANGED for image mode) ---

def load_images(image_path):
    img = cv.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    # Work on copies
    display_frame = img.copy()
    if display_frame.ndim == 2 or display_frame.shape[2] == 1:
        gray = display_frame if display_frame.ndim == 2 else cv.cvtColor(display_frame, cv.COLOR_BGR2GRAY)
    else:
        gray = cv.cvtColor(display_frame, cv.COLOR_BGR2GRAY)
    
    return display_frame, gray

def classify_face_all_models(face_resized):
    # This is a legacy function for the image mode's console output, kept for minimal disruption.
    
    results = classify_face_all_models_with_metrics(face_resized)
    
    return {
        "KNN": f"{results['KNN']['label']} (Conf:{results['KNN']['conf']:.2f}, MS:{results['KNN']['ms']:.1f})",
        "SVM": f"{results['SVM']['label']} (Conf:{results['SVM']['conf']:.2f}, MS:{results['SVM']['ms']:.1f})",
        "MiniX": f"{results['MiniX']['label']} (Conf:{results['MiniX']['conf']:.2f}, MS:{results['MiniX']['ms']:.1f})"
    }

def classify_image(image_path):
    display_frame, gray = load_images(image_path)

    # 检测人脸
    faces = dnn_detector.detect_faces(display_frame)

    if len(faces) == 0:
        print("[INFO] No faces detected in the image.")
    else:
        face_id = 0
        for (x, y, w, h) in faces:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(display_frame.shape[1], x + w), min(display_frame.shape[0], y + h)

            face_roi_gray = gray[y1:y2, x1:x2]
            if face_roi_gray.size == 0:
                continue

            face_resized = cv.resize(face_roi_gray, (48, 48))

            # 使用带有 metrics 的函数，但格式化为 image mode 的输出
            results = classify_face_all_models(face_resized)
            face_id += 1

            print(f"\n[FACE {face_id}]")
            for k, v in results.items():
                print(f"  {k}: {v}")
                
            # --- Drawing for image mode ---
            cv.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            text_lines = [
                f"KNN: {results['KNN']}",
                f"SVM: {results['SVM']}",
                f"MiniX: {results['MiniX']}",
            ]

            line_height = 18
            start_y = y1 - 5
            if start_y - line_height * len(text_lines) < 0:
                start_y = y1 + 20  

            for i, line in enumerate(text_lines):
                ty = start_y - i * line_height if start_y == y1 - 5 else start_y + i * line_height
                cv.putText(display_frame, line, (x1 + 5, ty),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)

    cv.imshow(windows_name, display_frame)
    print("[INFO] Press any key on the image window to close.")
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    
    load_models()

    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"[INFO] Running image mode on: {image_path}")
        classify_image(image_path)
    else:
        print("[INFO] No image path provided. Starting webcam mode...")
        video_classification()