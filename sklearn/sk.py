import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

# 데이터 준비
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]

fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]


fish_length = np.array(fish_length).reshape(-1, 1)
fish_weight = np.array(fish_weight)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(fish_length, fish_weight, test_size=0.3, random_state=42)

# 모델 생성 및 학습
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, c='black', label='Actual Data')
plt.scatter(X_test, y_pred_knn, c='green', alpha=0.7, label='KNN Prediction')
plt.scatter(X_test, y_pred_lr, c='purple', alpha=0.7, label='Linear Regression Prediction')

plt.xlabel('Fish Length (cm)')
plt.ylabel('Fish Weight (g)')
plt.title('Fish Weight Prediction (KNN vs Linear Regression)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

plt.show()
