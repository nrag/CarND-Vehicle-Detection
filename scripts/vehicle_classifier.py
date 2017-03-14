import numpy as np

from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

class VehicleClassifier:
    def __init__(self, car_features, noncar_features):
        rand_state = np.random.randint(0, 100)
        # Create an array stack of feature vectors
        X = np.vstack((car_features, noncar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        self.scaled_X = X_scaler.transform(X)
        self.y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.scaled_X, self.y, test_size=0.2, random_state=rand_state)
        self.svc = LinearSVC()

    def fit(self):
        self.svc.fit(self.X_train, self.y_train)
        return round(self.svc.score(self.X_test, self.y_test), 4)

    def predict(self, img_feature):
        return self.svc.predict(img_feature)


