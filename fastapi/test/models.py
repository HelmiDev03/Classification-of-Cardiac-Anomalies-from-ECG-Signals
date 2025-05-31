from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import wfdb
import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from glob import glob
from tqdm import tqdm
import pickle
import base64
import os
from fastapi import Form
import pandas as pd
import scipy

# Set up logging with explicit console output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Lifespan event handler to train or load models at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_models, global_results, global_X, global_y, training_status , le
    try:
        # Check if cached models and results exist
        if os.path.exists(MODELS_CACHE_PATH) and os.path.exists(RESULTS_CACHE_PATH):
            logger.info("Loading cached models and results...")
            training_status["status"] = "loading_cache"
            with open(MODELS_CACHE_PATH, "rb") as f:
                global_models = pickle.load(f)
            with open(RESULTS_CACHE_PATH, "rb") as f:
                global_results = pickle.load(f)
            logger.info("Cached models and results loaded successfully.")
            training_status["status"] = "completed"
        else:
            logger.info("Starting model training...")
            training_status["status"] = "in_progress"

            base_path = "mit-bih-arrhythmia-database-1.0.0"
            if not os.path.exists(base_path):
                raise FileNotFoundError(f"Data directory {base_path} not found")
            
            record_paths = glob(f"{base_path}/*.dat")
            if not record_paths:
                raise FileNotFoundError(f"No .dat files found in {base_path}")
            
            record_names = [p.split("/")[-1].replace(".dat", "") for p in record_paths]
            logger.info(f"Found {len(record_names)} records: {record_names}")
            
            for recordp in tqdm(record_names):
                record = wfdb.rdrecord(f"{recordp}", channels=[0])
                annotation = wfdb.rdann(f"{recordp}", 'atr')
                ecg = record.p_signal[:, 0]
                r_peaks = annotation.sample[1:]
                labels = annotation.symbol[1:]
                ecg = preprocess_ecg(ecg, record.fs)
                beats, labels = extract_beats(ecg, r_peaks, labels, record.fs)
                features = extract_features(beats, record.fs)
                global_X.extend(features)
                global_y.extend(labels)
            
            global_X = np.array(global_X)
            global_y = np.array(global_y)
            logger.info(f"Extracted {global_X.shape[0]} beats with {global_X.shape[1]} features each")

            X_train, X_test, y_train, y_test, le = prepare_data(global_X, global_y)
            global_models = train_models(X_train, y_train)
            
            global_results = {}
            for name, model in global_models.items():
                y_pred = model.predict(X_test)
                global_results[name] = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'Recall': recall_score(y_test, y_pred, average='weighted'),
                    'F1-Score': f1_score(y_test, y_pred, average='weighted'),
                    'Classification Report': classification_report(y_test, y_pred, zero_division=0)
                }
            
            # Save models and results to cache
            os.makedirs("cache", exist_ok=True)
            with open(MODELS_CACHE_PATH, "wb") as f:
                pickle.dump(global_models, f)
            with open(RESULTS_CACHE_PATH, "wb") as f:
                pickle.dump(global_results, f)
            logger.info("Models and results cached successfully.")
            logger.info("Model training completed.")
            training_status["status"] = "completed"
        
    except Exception as e:
        logger.error(f"Training or loading failed: {str(e)}")
        training_status["status"] = "failed"
        training_status["error"] = str(e)
        raise
    
    yield
    logger.info("Shutting down...")
    training_status["status"] = "shutdown"

# Initialize FastAPI app
app = FastAPI(
    title="ECG Analysis API",
    version="1.0.0",
    lifespan=lifespan  # Will be set below
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global variables to store models and results
global_models = {}
global_results = {}
global_X = []
global_y = []
training_status = {"status": "not_started", "error": None}
le = None

# Paths for caching models and results
MODELS_CACHE_PATH = "cache/models.pkl"
RESULTS_CACHE_PATH = "cache/results.pkl"

# Preprocessing and feature extraction functions (unchanged)
def preprocess_ecg(ecg_signal, fs):
    nyquist = 0.5 * fs
    low = 0.5 / nyquist
    high = 15.0 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, ecg_signal)
    notch_freq = 50.0
    Q = 35.0
    b, a = signal.iirnotch(notch_freq/nyquist, Q)
    filtered = signal.filtfilt(b, a, filtered)
    normalized = (filtered - np.mean(filtered)) / np.std(filtered)
    return normalized

def extract_beats(signal, ann_samples, ann_symbols, fs, before=0.25, after=0.4):
    beats = []
    labels = []
    before_samples = int(before * fs)
    after_samples = int(after * fs)
    for i, (sample, symbol) in enumerate(zip(ann_samples, ann_symbols)):
        start = sample - before_samples
        end = sample + after_samples
        if start >= 0 and end < len(signal):
            beat = signal[start:end]
            beats.append(beat)
            labels.append(symbol)
    return np.array(beats), np.array(labels)

def ricker_wavelet(points, a):
    A = 2 / (np.sqrt(3 * a) * (np.pi ** 0.25))
    wsq = a ** 2
    t = np.linspace(-points / 2, points / 2, points)
    mod = 1 - (t ** 2) / wsq
    gauss = np.exp(-(t ** 2) / (2 * wsq))
    return A * mod * gauss

def extract_features(beats, fs):
    features = []
    for beat in beats:
        beat_features = [
            np.mean(beat),
            np.std(beat),
            np.min(beat),
            np.max(beat)
        ]
        wavelet = ricker_wavelet(points=len(beat), a=4.0)
        conv_result = signal.convolve(beat, wavelet, mode='same')
        energy = np.sum(conv_result ** 2)
        max_val = np.max(np.abs(conv_result))
        mean_val = np.mean(conv_result)
        beat_features.extend([energy, max_val, mean_val])
        middle = len(beat) // 2
        qrs = beat[middle-10:middle+10]
        beat_features.append(np.ptp(qrs))
        beat_features.append(len(qrs)/fs)
        features.append(beat_features)
    return features

def prepare_data(X, y, test_size=0.2, random_state=42):

    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size,
        shuffle=True,
        random_state=random_state)
    return X_train, X_test, y_train, y_test, le

def train_models(X_train, y_train):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale'),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

# Lifespan event handler to train or load models at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_models, global_results, global_X, global_y, training_status ,le
    try:
        # Check if cached models and results exist
        if os.path.exists(MODELS_CACHE_PATH) and os.path.exists(RESULTS_CACHE_PATH):
            logger.info("Loading cached models and results...")
            training_status["status"] = "loading_cache"
            with open(MODELS_CACHE_PATH, "rb") as f:
                global_models = pickle.load(f)
            with open(RESULTS_CACHE_PATH, "rb") as f:
                global_results = pickle.load(f)
            logger.info("Cached models and results loaded successfully.")
            training_status["status"] = "completed"
        else:
            logger.info("Starting model training...")
            training_status["status"] = "in_progress"

            base_path = "mit-bih-arrhythmia-database-1.0.0"
            if not os.path.exists(base_path):
                raise FileNotFoundError(f"Data directory {base_path} not found")
            
            record_paths = glob(f"{base_path}/*.dat")
            if not record_paths:
                raise FileNotFoundError(f"No .dat files found in {base_path}")
            
            record_names = [p.split("/")[-1].replace(".dat", "") for p in record_paths]
            logger.info(f"Found {len(record_names)} records: {record_names}")
            
            for recordp in tqdm(record_names):
                record = wfdb.rdrecord(f"{base_path}/{recordp}", channels=[0])
                annotation = wfdb.rdann(f"{base_path}/{recordp}", 'atr')
                ecg = record.p_signal[:, 0]
                r_peaks = annotation.sample[1:]
                labels = annotation.symbol[1:]
                ecg = preprocess_ecg(ecg, record.fs)
                beats, labels = extract_beats(ecg, r_peaks, labels, record.fs)
                features = extract_features(beats, record.fs)
                global_X.extend(features)
                global_y.extend(labels)
            
            global_X = np.array(global_X)
            global_y = np.array(global_y)
            logger.info(f"Extracted {global_X.shape[0]} beats with {global_X.shape[1]} features each")

            X_train, X_test, y_train, y_test, le = prepare_data(global_X, global_y)
            global_models = train_models(X_train, y_train)
            
            global_results = {}
            for name, model in global_models.items():
                y_pred = model.predict(X_test)
                global_results[name] = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'Recall': recall_score(y_test, y_pred, average='weighted'),
                    'F1-Score': f1_score(y_test, y_pred, average='weighted'),
                    'Classification Report': classification_report(y_test, y_pred, zero_division=0)
                }
            
            # Save models and results to cache
            os.makedirs("cache", exist_ok=True)
            with open(MODELS_CACHE_PATH, "wb") as f:
                pickle.dump(global_models, f)
            with open(RESULTS_CACHE_PATH, "wb") as f:
                pickle.dump(global_results, f)
            logger.info("Models and results cached successfully.")
            logger.info("Model training completed.")
            training_status["status"] = "completed"
        
    except Exception as e:
        logger.error(f"Training or loading failed: {str(e)}")
        training_status["status"] = "failed"
        training_status["error"] = str(e)
        raise
    
    yield
    logger.info("Shutting down...")
    training_status["status"] = "shutdown"

app.lifespan = lifespan







base_path = "mit-bih-arrhythmia-database-1.0.0"
if not os.path.exists(base_path):
    raise FileNotFoundError(f"Data directory {base_path} not found")

record_paths = glob(f"{base_path}/*.dat")
if not record_paths:
    raise FileNotFoundError(f"No .dat files found in {base_path}")

record_names = [p.split("/")[-1].replace(".dat", "") for p in record_paths]
logger.info(f"Found {len(record_names)} records: {record_names}")

for recordp in tqdm(record_names):
    record = wfdb.rdrecord(f"{recordp}", channels=[0])
    annotation = wfdb.rdann(f"{recordp}", 'atr')
    ecg = record.p_signal[:, 0]
    r_peaks = annotation.sample[1:]
    labels = annotation.symbol[1:]
    ecg = preprocess_ecg(ecg, record.fs)
    beats, labels = extract_beats(ecg, r_peaks, labels, record.fs)
    features = extract_features(beats, record.fs)
    global_X.extend(features)
    global_y.extend(labels)

global_X = np.array(global_X)
global_y = np.array(global_y)
logger.info(f"Extracted {global_X.shape[0]} beats with {global_X.shape[1]} features each")

X_train, X_test, y_train, y_test, le = prepare_data(global_X, global_y)








@app.get("/metrics/", summary="Get model metrics")
async def get_metrics():
    if not global_results:
        return JSONResponse(
            content={"error": "Metrics not available", "training_status": training_status},
            status_code=500
        )
    return {"results": global_results}

@app.get("/status", summary="Check training status")
async def get_status():
    return {"training_status": training_status}





@app.get("/analyze", summary="Analyze ECG with KNN model")
async def analyze_knn():
    """
    Dummy endpoint to predict with KNN model using fake data
    """
    # Fake input data
    fake_features = [[0.5, 0.2, 0.1, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]]
    # Use the loaded knn_model if available, else return dummy result
    if global_models['KNN']:
        try:
            prediction = global_models['KNN'].predict(fake_features)
            result = int(prediction[0])
        except Exception as e:
            result = f"Prediction failed: {str(e)}"
    else:
        result = "KNN model not loaded"
    return JSONResponse(content={
        "model": "knn",
        "prediction": result,
        "input": fake_features
    })


def extract_beats1(signal, ann_samples,  fs, before=0.25, after=0.4):
    """
    Extract beats centered at R-peaks
    before: seconds before R-peak
    after: seconds after R-peak
    """
    beats = []
    before_samples = int(before * fs)
    after_samples = int(after * fs)
    
    for i, sample in enumerate(ann_samples):
        # Only consider normal and some abnormal beats for this example
        start = sample - before_samples
        end = sample + after_samples
        
        # Check boundaries
        if start >= 0 and end < len(signal):
            beat = signal[start:end]
            beats.append(beat)
            
    
    return np.array(beats)



@app.post("/predict", summary="Analyze ECG with Decision Tree model")
async def predict(
    dat: UploadFile = File(...),
    hea: UploadFile = File(...),
    model_name: str = Form(...)
):
    # Save uploaded files to the userfiles directory
    userfiles_dir = os.path.join(os.path.dirname(__file__), "userfiles")
    os.makedirs(userfiles_dir, exist_ok=True)

    base_filename = os.path.splitext(dat.filename)[0]  # e.g., "sample" from "sample.dat"

    dat_path = os.path.join(userfiles_dir, f"{base_filename}.dat")
    hea_path = os.path.join(userfiles_dir, f"{base_filename}.hea")

    with open(dat_path, "wb") as f:
        f.write(await dat.read())
    with open(hea_path, "wb") as f:
        f.write(await hea.read())

    # Use base path (without extension)
    base_path = os.path.join(userfiles_dir, base_filename)

    # Load with wfdb using just the base path
    record = wfdb.rdrecord(base_path)
    

    print(f"Loaded record: {record}")



    ecg = record.p_signal[:, 0]  # Use first ECG lead
    fs = record.fs
    beat=pd.Series(ecg)
    
    r_peaks1,meta=scipy.signal.find_peaks(beat,height=0.8)
    ecg = preprocess_ecg(ecg, fs)
    beats = extract_beats1(ecg, r_peaks, fs)
    features = extract_features(beats,fs)

    if model_name=='KNN':
        try:
            OutputOfModel = global_models['KNN'].predict(features)
            
        except Exception as e:
            result = f"Prediction failed: {str(e)}"
    elif model_name=='Decision Tree':
        try:
            OutputOfModel = global_models['Decision Tree'].predict(features)
        except Exception as e:
            result = f"Prediction failed: {str(e)}"
    elif model_name=='Random Forest':
        try:
            OutputOfModel = global_models['Random Forest'].predict(features)
        except Exception as e:
            result = f"Prediction failed: {str(e)}"

    elif model_name=='SVM':
        try:
            OutputOfModel = global_models['SVM'].predict(features)
        except Exception as e:
            result = f"Prediction failed: {str(e)}"                        
    
     
    mapped_results = le.inverse_transform(OutputOfModel)
    used = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
    mapped_results = [x for x in mapped_results if x  in used]
    RESULT = {
        "OutputOfModel": mapped_results
    }
    return JSONResponse(content={
            "model": model_name,
            "result": RESULT
        })
    



    
 


#set PATH=%PATH%;C:\Users\helmi\AppData\Roaming\Python\Python313\Scripts
#uvicorn models:app