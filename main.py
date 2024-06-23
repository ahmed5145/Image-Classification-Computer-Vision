import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler

# Prepare data
input_dir = 'C:\\Users\\hussah01\\Desktop\\CS Projects\\Image Classification Computer Vision\\Chess'
categories = ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']
data = []
labels = []
target_shape = (64, 64)  # Target image resolution

# Supported image extensions
supported_extensions = ('.jpg', '.jpeg', '.png')

for category_idx, category in enumerate(categories):
    category_path = os.path.join(input_dir, category)
    for file in os.listdir(category_path):
        if file.lower().endswith(supported_extensions):  # Check for supported image extensions
            img_path = os.path.join(category_path, file)
            img = imread(img_path)
            img = resize(img, target_shape)  # Resize image
            img = img.astype(np.float32)  # Convert to float32
            data.append(img.flatten())
            labels.append(category_idx)

data_array = np.array(data)
labels_array = np.array(labels)

# Normalize data
scaler = StandardScaler()
data_array = scaler.fit_transform(data_array)

# Train / Test Split
x_train, x_test, y_train, y_test = train_test_split(data_array, labels_array, test_size=0.2, shuffle=True, stratify=labels_array)

# Train classifier
classifier = SVC()

parameters = [{'gamma': [0.1, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(x_train, y_train)

# Test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print(f'{score*100:.2f}% of samples were correctly classified')

# Save the model
pickle.dump(best_estimator, open('./model.p', 'wb'))