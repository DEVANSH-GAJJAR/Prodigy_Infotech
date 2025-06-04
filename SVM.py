import os 
import cv2
import numpy as np
from sklearn.svm import SVC 
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import random

#THE HOG FEATURE 'S EXTRACTION 

def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys',
                      visualize=True)
    return features

#LOADING AND PREPROCESSING OF THE DATA 

def load_data(data_dir, label, max_images=1000):
    features = []
    labels = []
    images = os.listdir(data_dir)
    random.shuffle(images)
    for img_name in images[:max_images]:
        img_path = os.path.join(data_dir, img_name)
        try:
            image = cv2.imread(img_path)
            image = cv2.resize(image, (64, 64))
            hog_feat = extract_hog_features(image)
            features.append(hog_feat)
            labels.append(label)
        except:
            continue  #LEAVE IT THESE THINGS (SKIP IT !!)
    return features, labels

#DEFINING OF THE PATHS 
cat_dir = r"E:\PRODIGY-INFOTECH\SVMdataset\test_set\test_set\cats"
dog_dir = r"E:\PRODIGY-INFOTECH\SVMdataset\test_set\test_set\dogs"


#LOADING THE DATA 
cat_features, cat_labels = load_data(cat_dir, label=0, max_images=1000)
dog_features, dog_labels = load_data(dog_dir, label=1, max_images=1000)

# COMBINING OF THE DATA 
X = np.array(cat_features + dog_features)
y = np.array(cat_labels + dog_labels)

# SPLITTING OF THE DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TRAINING YOUR SVM 
print("Training the model...")
model = SVC(kernel='linear')  
model.fit(X_train, y_train)

# EVALUATYION
print("Evaluating the model...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

