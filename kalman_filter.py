import numpy as np


## mostly copied from https://www.geeksforgeeks.org/python/kalman-filter-in-python/
class KalmanFilter:
    def __init__(self, initial_position, dt = 0.5):
        ### assume constantly velocity
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        ### no control input, this is for obstacles
        self.B = np.zeros((4, 2))
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 0.5
        self.x = np.array([initial_position[0], initial_position[1], 0.0, 0.0])
        self.P = np.eye(4) * 1.0

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x
    
    def get_position(self):
        return self.x[:2]
    
    def get_velocity(self):
        return self.x[2:]