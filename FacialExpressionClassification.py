import cv2 as cv
import sys
import joblib
import numpy as np
from skimage.feature import local_binary_pattern, hog
from tensorflow.keras.models import load_model
from FaceDetector.DNN_FaceDetector import DnnDetector

CAM_INDEX = 0  # Change this if you have multiple cameras

windows_name = "Facial Emotion Recognition"

TB_Model = "Model 0:KNN 1:SVM 2:Mini-Xception"
TB_Model_max_value = 2
TB_Model_default = 2

model = TB_Model_default

FACE_DETECTOR_ROOT = "FaceDetector"
dnn_detector = DnnDetector(root = FACE_DETECTOR_ROOT)

CLASSES = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

def load_models():
    global knn_model, svm_model, lbp_scaler, hog_scaler, mini_x_model

    knn_model_path = "Model/knn_lbp_model.joblib"
    svm_model_path = "Model/svm_hog_model.joblib"
    lbp_scaler_path = "Model/scaler_lbp.joblib"
    hog_scaler_path = "Model/scaler_hog.joblib"
    mini_x_path = "Model/best_model.keras"

    try:
        knn_model = joblib.load(knn_model_path)
        print("[INFO] KNN model loaded:", knn_model_path)
    except Exception as e:
        print("[WARN] Fail to load KNN model:", e)

    try:
        svm_model = joblib.load(svm_model_path)
        print("[INFO] SVM model loaded:", svm_model_path)
    except Exception as e:
        print("[WARN] Fail to load SVM model:", e)

    try:
        lbp_scaler = joblib.load(lbp_scaler_path)
        print("[INFO] LBP Scaler loaded:", lbp_scaler_path)
    except Exception as e:
        print("[WARN] Fail to load LBP scaler:", e)

    try:
        hog_scaler = joblib.load(hog_scaler_path)
        print("[INFO] HOG Scaler loaded:", hog_scaler_path)
    except Exception as e:
        print("[WARN] Fail to load HOG scaler:", e)

    try:
        mini_x_model = load_model(mini_x_path)
        print("[INFO] Mini-Xception model loaded:", mini_x_path)
    except Exception as e:
        print("[WARN] Fail to load Mini-Xception model:", e)

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


def CreateGUI():
    
    global model
    cv.namedWindow(windows_name, cv.WINDOW_NORMAL)

    cv.createTrackbar(TB_Model, windows_name, int(model), TB_Model_max_value, process_display_callback)

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
    

def predict_emotion_knn(lbp_features):
    if knn_model is None or lbp_scaler is None:
        return "KNN Not Loaded"
    feature = lbp_features.reshape(1,-1).astype("float32")
    feature_scaled = lbp_scaler.transform(feature)
    y_pred = knn_model.predict(feature_scaled)[0]
    return CLASSES.get(int(y_pred), str(y_pred))

def predict_emotion_svm(hog_features):
    if svm_model is None or hog_scaler is None:
        return "SVM-NotLoaded"

    feature = hog_features.reshape(1, -1).astype("float32")
    feature_scaled = hog_scaler.transform(feature)
    y_pred = svm_model.predict(feature_scaled)[0]
    return CLASSES.get(int(y_pred), str(y_pred))

def predict_emotion_minix(face_gray):
    if mini_x_model is None:
        return "MiniX-NotLoaded"

    face_norm = face_gray.astype("float32") / 255.0
    face_input = face_norm.reshape(1, 48, 48, 1)
    probs = mini_x_model.predict(face_input, verbose=0)[0]
    cls_id = int(np.argmax(probs))
    return CLASSES.get(cls_id, str(cls_id))
               

def classify_face_all_models(face_resized):

    if (knn_model is not None) and (lbp_scaler is not None):
        lbp_features = LBP_Features_extract(face_resized)
        label_knn = predict_emotion_knn(lbp_features)
    else:
        label_knn = "KNN N/A"

    if (svm_model is not None) and (hog_scaler is not None):
        hog_features = HOG_Features_extract(face_resized)
        label_svm = predict_emotion_svm(hog_features)
    else:
        label_svm = "SVM N/A"

    label_minix = predict_emotion_minix(face_resized)

    return {
        "KNN": label_knn,
        "SVM": label_svm,
        "MiniX": label_minix
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

            results = classify_face_all_models(face_resized)
            face_id += 1

            print(f"\n[FACE {face_id}]")
            for k, v in results.items():
                print(f"  {k}: {v}")

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

def process_display():
    global frame_in, frame_gray
    
    # Display the frame
    display_frame = frame_in.copy()
    gray = frame_gray
    

    faces = dnn_detector.detect_faces(display_frame)

    for (x, y, w, h) in faces:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(display_frame.shape[1], x + w), min(display_frame.shape[0], y + h)

        face_roi_gray = gray[y1:y2, x1:x2]
        if face_roi_gray.size == 0:
            continue

        face_resized = cv.resize(face_roi_gray, (48, 48))

        if model == 0:
            lbp_features = LBP_Features_extract(face_resized)
            label = predict_emotion_knn(lbp_features)
        elif model == 1:
            hog_features = HOG_Features_extract(face_resized)
            label = predict_emotion_svm(hog_features)
        else:
            label = predict_emotion_minix(face_resized)

        cv.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv.rectangle(display_frame, (x1, y1 - 20), (x1 + 160, y1), (50, 50, 50), -1)
        cv.putText(display_frame, label, (x1 + 5, y1 - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)



    cv.imshow(windows_name, display_frame)


def process_display_callback(value):
    global model
    TB_Model_value = cv.getTrackbarPos(TB_Model, windows_name)
    model = TB_Model_value
    model_names = {0:"KNN", 1:"SVM", 2:"Mini-Xception"}
    model_name = model_names.get(model,str(model))
    title_text = (f"{windows_name} | Model ={model_name}")
    cv.setWindowTitle(windows_name,title_text)

    process_display()


def video_classification():
    initialize_stream()
    grab_preprocess()
    CreateGUI()
    process_display_callback(0)

    while True:
        if not grab_preprocess():
            break

        process_display()

        if cv.waitKey(5) >=0:
            break

    cap.release()
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