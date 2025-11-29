import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 한글 폰트 설정 (Windows/Linux/Mac 환경별 폰트)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

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

X_train, X_test, y_train, y_test = train_test_split(fish_length, fish_weight, test_size=0.3, random_state=42)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("="*80)
print("KNN Regression vs Linear Regression 성능 비교")
print("="*80)
print(f"\n{'모델':<20} {'MSE':<15} {'RMSE':<15} {'R² Score':<15}")
print("-"*80)
print(f"{'KNN Regressor':<20} {mean_squared_error(y_test, y_pred_knn):<15.2f} {np.sqrt(mean_squared_error(y_test, y_pred_knn)):<15.2f} {r2_score(y_test, y_pred_knn):<15.4f}")
print(f"{'Linear Regression':<20} {mean_squared_error(y_test, y_pred_lr):<15.2f} {np.sqrt(mean_squared_error(y_test, y_pred_lr)):<15.2f} {r2_score(y_test, y_pred_lr):<15.4f}")

print("\n" + "="*80)
print("큰 값 예측 성능 비교 (무게 > 700g)")
print("="*80)
print(f"{'실제 길이':<12} {'실제 무게':<12} {'KNN 예측':<12} {'Linear 예측':<15} {'KNN 오차':<12} {'Linear 오차'}")
print("-"*80)

large_fish = [(X_test[i][0], y_test[i], y_pred_knn[i], y_pred_lr[i]) 
              for i in range(len(y_test)) if y_test[i] > 700]
large_fish.sort(key=lambda x: x[1], reverse=True)

for length, actual, knn_pred, lr_pred in large_fish:
    knn_error = abs(actual - knn_pred)
    lr_error = abs(actual - lr_pred)
    print(f"{length:<12.1f} {actual:<12.1f} {knn_pred:<12.1f} {lr_pred:<15.1f} {knn_error:<12.1f} {lr_error:.1f}")

print("\n결론:")
print("- Dataset 범위 내 데이터: 두 모델 모두 괜찮은 성능")
print("- Dataset 범위 밖 큰 값: KNN은 이웃의 평균만 사용 → 과소평가")
print("- Dataset 범위 밖 큰 값: Linear Regression은 외삽 가능 → 더 정확")

# ============================================================================
# 시각화 (Visualization)
# ============================================================================

# Figure 생성 (2x2 서브플롯)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Fish Weight Prediction: KNN vs Linear Regression', fontsize=18, fontweight='bold', y=0.995)

# 1. 원본 데이터 및 Train/Test 분할 시각화
ax1 = axes[0, 0]
ax1.scatter(X_train, y_train, c='#3498db', alpha=0.6, s=80, label='Train Data', edgecolors='black', linewidths=1)
ax1.scatter(X_test, y_test, c='#e74c3c', alpha=0.6, s=80, label='Test Data', edgecolors='black', linewidths=1)
ax1.set_xlabel('Fish Length (cm)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Fish Weight (g)', fontsize=12, fontweight='bold')
ax1.set_title('Original Data Distribution (Train/Test Split)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')

# 2. 예측 결과 비교 (Actual vs Predicted)
ax2 = axes[0, 1]
# 전체 데이터 범위에서 예측 라인 그리기
x_range = np.linspace(fish_length.min(), fish_length.max(), 300).reshape(-1, 1)
y_knn_range = knn.predict(x_range)
y_lr_range = lr.predict(x_range)

ax2.scatter(X_test, y_test, c='black', alpha=0.7, s=120, label='Actual', marker='*', edgecolors='white', linewidths=1.5, zorder=3)
ax2.scatter(X_test, y_pred_knn, c='#2ecc71', alpha=0.6, s=80, label='KNN Prediction', edgecolors='black', linewidths=1)
ax2.scatter(X_test, y_pred_lr, c='#9b59b6', alpha=0.6, s=80, label='Linear Regression Prediction', edgecolors='black', linewidths=1)
ax2.plot(x_range, y_knn_range, 'g--', linewidth=2, label='KNN Fit Line', alpha=0.7)
ax2.plot(x_range, y_lr_range, 'm-', linewidth=2, label='Linear Fit Line', alpha=0.7)
ax2.set_xlabel('Fish Length (cm)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Fish Weight (g)', fontsize=12, fontweight='bold')
ax2.set_title('Prediction Results Comparison', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, alpha=0.3, linestyle='--')

# 3. 잔차 플롯 (Residual Plot)
ax3 = axes[1, 0]
residuals_knn = y_test - y_pred_knn
residuals_lr = y_test - y_pred_lr

ax3.scatter(y_pred_knn, residuals_knn, c='#2ecc71', alpha=0.6, s=80, label='KNN Residuals', edgecolors='black', linewidths=1)
ax3.scatter(y_pred_lr, residuals_lr, c='#9b59b6', alpha=0.6, s=80, label='Linear Regression Residuals', edgecolors='black', linewidths=1)
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error Line')
ax3.set_xlabel('Predicted Weight (g)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
ax3.set_title('Residual Plot (Error Distribution)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, linestyle='--')

# 4. 모델 성능 비교 (Model Performance Comparison)
ax4 = axes[1, 1]
metrics = ['MSE', 'RMSE', 'R² Score']
knn_metrics = [
    mean_squared_error(y_test, y_pred_knn),
    np.sqrt(mean_squared_error(y_test, y_pred_knn)),
    r2_score(y_test, y_pred_knn)
]
lr_metrics = [
    mean_squared_error(y_test, y_pred_lr),
    np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    r2_score(y_test, y_pred_lr)
]

x_pos = np.arange(len(metrics))
width = 0.35

bars1 = ax4.bar(x_pos - width/2, knn_metrics, width, 
                label='KNN', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax4.bar(x_pos + width/2, lr_metrics, width, 
                label='Linear Regression', color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1.5)

# 값 표시
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax4.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
ax4.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(metrics, fontsize=11)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
plt.savefig('/home/hyuksu/projects/ml/sklearn/model_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n{'='*80}")
print("시각화 완료! 그래프가 '/home/hyuksu/projects/ml/sklearn/model_comparison.png'에 저장되었습니다.")
print(f"{'='*80}")
plt.show()
