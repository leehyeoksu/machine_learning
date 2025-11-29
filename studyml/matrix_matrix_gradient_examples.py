"""
행렬-행렬 미분 실전 예제
Matrix-to-Matrix Derivatives with Concrete Examples

이 파일은 행렬 미분의 헷갈리는 부분을 명확하게 보여줍니다.
"""

import numpy as np

# ============================================================================
# 유틸리티 함수: Numerical Gradient
# ============================================================================

def numerical_gradient_matrix(f, A, eps=1e-5):
    """
    행렬 A에 대한 스칼라 함수 f의 numerical gradient 계산
    
    Parameters:
    - f: function that takes matrix A and returns scalar
    - A: numpy array of any shape
    - eps: small perturbation for finite difference
    
    Returns:
    - grad: same shape as A, containing ∂f/∂A
    """
    grad = np.zeros_like(A)
    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            # A의 (i,j) element를 eps만큼 증가
            A_plus = A.copy()
            A_plus[i, j] += eps
            f_plus = f(A_plus)
            
            # A의 (i,j) element를 eps만큼 감소
            A_minus = A.copy()
            A_minus[i, j] -= eps
            f_minus = f(A_minus)
            
            # Finite difference
            grad[i, j] = (f_plus - f_minus) / (2 * eps)
    
    return grad


def compare_gradients(analytical, numerical, name="Gradient"):
    """두 gradient를 비교하고 결과 출력"""
    print(f"\n{'='*60}")
    print(f"{name} 비교")
    print(f"{'='*60}")
    print(f"Analytical gradient:\n{analytical}\n")
    print(f"Numerical gradient:\n{numerical}\n")
    error = np.max(np.abs(analytical - numerical))
    print(f"Max error: {error:.2e}")
    
    if error < 1e-5:
        print("✅ PASS: Gradient가 정확합니다!")
    else:
        print("❌ FAIL: Gradient에 문제가 있습니다.")
    print(f"{'='*60}\n")


# ============================================================================
# 예제 1: tr(AB) - 가장 기본적인 경우
# ============================================================================

print("\n" + "="*70)
print("예제 1: ∂tr(AB)/∂A = B^T")
print("="*70)

# Setup
A = np.random.randn(3, 2)  # 3x2 행렬
B = np.random.randn(2, 3)  # 2x3 행렬 (고정)

print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")
print(f"AB shape: {(A @ B).shape}")

# 함수 정의: f(A) = tr(AB)
def f1(A):
    return np.trace(A @ B)

# Analytical gradient: ∂tr(AB)/∂A = B^T
grad_analytical = B.T

# Numerical gradient
grad_numerical = numerical_gradient_matrix(f1, A)

compare_gradients(grad_analytical, grad_numerical, "∂tr(AB)/∂A")


# ============================================================================
# 예제 2: tr(A^T B) - 전치 위치 중요!
# ============================================================================

print("\n" + "="*70)
print("예제 2: ∂tr(A^T B)/∂A = B")
print("="*70)

# Setup: A와 B가 같은 크기
A = np.random.randn(3, 2)
B = np.random.randn(3, 2)  # 같은 크기!

print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")

# 함수 정의: f(A) = tr(A^T B)
def f2(A):
    return np.trace(A.T @ B)

# Analytical gradient: ∂tr(A^T B)/∂A = B
grad_analytical = B

# Numerical gradient
grad_numerical = numerical_gradient_matrix(f2, A)

compare_gradients(grad_analytical, grad_numerical, "∂tr(A^T B)/∂A")


# ============================================================================
# 예제 3: Frobenius Norm - ||A||_F^2
# ============================================================================

print("\n" + "="*70)
print("예제 3: ∂||A||_F^2 / ∂A = 2A")
print("="*70)

A = np.random.randn(2, 3)

print(f"A shape: {A.shape}")
print(f"||A||_F = {np.linalg.norm(A, 'fro'):.4f}")

# 함수 정의: f(A) = ||A||_F^2 = tr(A^T A)
def f3(A):
    return np.sum(A ** 2)  # 또는 np.trace(A.T @ A)

# Analytical gradient
grad_analytical = 2 * A

# Numerical gradient
grad_numerical = numerical_gradient_matrix(f3, A)

compare_gradients(grad_analytical, grad_numerical, "∂||A||_F^2 / ∂A")


# ============================================================================
# 예제 4: ||AX - B||_F^2 - Linear Regression with matrices
# ============================================================================

print("\n" + "="*70)
print("예제 4: ∂||AX - B||_F^2 / ∂A = 2(AX - B)X^T")
print("="*70)

# Setup
A = np.random.randn(3, 4)  # 우리가 최적화할 행렬 (weight matrix)
X = np.random.randn(4, 5)  # 입력 데이터 (고정)
B = np.random.randn(3, 5)  # 목표 출력 (고정)

print(f"A shape: {A.shape}")
print(f"X shape: {X.shape}")
print(f"B shape: {B.shape}")
print(f"AX shape: {(A @ X).shape}")

# 함수 정의: f(A) = ||AX - B||_F^2
def f4(A):
    residual = A @ X - B
    return np.sum(residual ** 2)

# Analytical gradient: ∂||AX - B||_F^2 / ∂A = 2(AX - B)X^T
residual = A @ X - B
grad_analytical = 2 * residual @ X.T

# Numerical gradient
grad_numerical = numerical_gradient_matrix(f4, A)

compare_gradients(grad_analytical, grad_numerical, "∂||AX - B||_F^2 / ∂A")


# ============================================================================
# 예제 5: tr(AXA^T) - 대칭 형태
# ============================================================================

print("\n" + "="*70)
print("예제 5: ∂tr(AXA^T)/∂A = AX^T + AX (X가 대칭이면 2AX)")
print("="*70)

# Setup
A = np.random.randn(3, 3)
X = np.random.randn(3, 3)
X = (X + X.T) / 2  # X를 대칭 행렬로 만들기

print(f"A shape: {A.shape}")
print(f"X shape: {X.shape}")
print(f"X is symmetric: {np.allclose(X, X.T)}")

# 함수 정의: f(A) = tr(AXA^T)
def f5(A):
    return np.trace(A @ X @ A.T)

# Analytical gradient
# X가 대칭이면: ∂tr(AXA^T)/∂A = 2AX
grad_analytical = 2 * A @ X

# Numerical gradient
grad_numerical = numerical_gradient_matrix(f5, A)

compare_gradients(grad_analytical, grad_numerical, "∂tr(AXA^T)/∂A (X symmetric)")


# ============================================================================
# 예제 6: Element별 계산 vs 공식 (작은 예제로 직접 확인)
# ============================================================================

print("\n" + "="*70)
print("예제 6: Element별 계산으로 공식 유도 확인")
print("="*70)

# 간단한 2x2 예제
A = np.array([[1.0, 2.0],
              [3.0, 4.0]])

X = np.array([[5.0, 6.0],
              [7.0, 8.0]])

print("A =")
print(A)
print("\nX =")
print(X)

# f(A) = tr(AX) 계산
AX = A @ X
f_val = np.trace(AX)
print(f"\nAX =")
print(AX)
print(f"\ntr(AX) = {f_val}")

# Element별로 직접 계산
print("\n--- Element별 미분 계산 ---")
print("f = tr(AX) = (AX)[0,0] + (AX)[1,1]")
print("  = (A[0,0]*X[0,0] + A[0,1]*X[1,0]) + (A[1,0]*X[0,1] + A[1,1]*X[1,1])")
print(f"  = ({A[0,0]}*{X[0,0]} + {A[0,1]}*{X[1,0]}) + ({A[1,0]}*{X[0,1]} + {A[1,1]}*{X[1,1]})")
print(f"  = {A[0,0]*X[0,0] + A[0,1]*X[1,0]} + {A[1,0]*X[0,1] + A[1,1]*X[1,1]}")
print(f"  = {f_val}")

print("\nElement별 편미분:")
print(f"∂f/∂A[0,0] = X[0,0] = {X[0,0]}")
print(f"∂f/∂A[0,1] = X[1,0] = {X[1,0]}")
print(f"∂f/∂A[1,0] = X[0,1] = {X[0,1]}")
print(f"∂f/∂A[1,1] = X[1,1] = {X[1,1]}")

grad_manual = np.array([[X[0,0], X[1,0]],
                        [X[0,1], X[1,1]]])

print("\n수동으로 만든 gradient:")
print(grad_manual)

print("\nX^T (공식 사용):")
print(X.T)

print("\n비교:")
print(f"수동 계산 == X^T: {np.allclose(grad_manual, X.T)}")


# ============================================================================
# 예제 7: 다른 크기 행렬 (2x4 @ 4x2)
# ============================================================================

print("\n" + "="*70)
print("예제 7: 다른 크기 행렬 - A(2x4) @ X(4x2)")
print("="*70)

A = np.random.randn(2, 4)  # 2x4
X = np.random.randn(4, 2)  # 4x2

print(f"A shape: {A.shape}")
print(f"X shape: {X.shape}")
print(f"AX shape: {(A @ X).shape}")

# f(A) = tr(AX)
def f7(A):
    return np.trace(A @ X)

# Analytical: ∂tr(AX)/∂A = X^T
# X^T는 2x4가 되어야 A와 같은 크기!
print(f"X^T shape: {X.T.shape}")  # 2x4 ✓

grad_analytical = X.T

# Numerical
grad_numerical = numerical_gradient_matrix(f7, A)

compare_gradients(grad_analytical, grad_numerical, "∂tr(AX)/∂A (different sizes)")


# ============================================================================
# 예제 8: Chain Rule - Neural Network Weight Gradient
# ============================================================================

print("\n" + "="*70)
print("예제 8: Neural Network - ∂L/∂W where L = ||WX - Y||^2")
print("="*70)

# Setup: W는 weight matrix, X는 input, Y는 target
W = np.random.randn(3, 4)  # Weight matrix (output_dim x input_dim)
X = np.random.randn(4, 10)  # Input batch (input_dim x batch_size)
Y = np.random.randn(3, 10)  # Target batch (output_dim x batch_size)

print(f"W shape: {W.shape}")
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")

# Loss function
def loss(W):
    predictions = W @ X
    return 0.5 * np.sum((predictions - Y) ** 2)

# Analytical gradient: ∂L/∂W = (WX - Y) @ X^T
predictions = W @ X
residual = predictions - Y
grad_analytical = residual @ X.T

# Numerical gradient
grad_numerical = numerical_gradient_matrix(loss, W)

compare_gradients(grad_analytical, grad_numerical, "∂L/∂W (Neural Network)")


# ============================================================================
# 요약
# ============================================================================

print("\n" + "="*70)
print("요약: 핵심 공식들")
print("="*70)

summary = """
✅ 행렬 미분의 핵심 원칙:
   - Gradient의 shape = 미분 대상의 shape
   - ∂f/∂A ∈ R^(m×n) if A ∈ R^(m×n)

✅ 자주 쓰이는 공식:
   1. ∂tr(AB)/∂A = B^T
   2. ∂tr(A^T B)/∂A = B
   3. ∂||A||_F^2/∂A = 2A
   4. ∂||AX - B||_F^2/∂A = 2(AX - B)X^T
   5. ∂tr(AXA^T)/∂A = A(X + X^T)  [X 대칭이면 2AX]

✅ 검증 방법:
   - 항상 numerical gradient로 확인!
   - Shape가 맞는지 먼저 체크!
   - 간단한 2x2 예제로 element별 계산해보기!

✅ ML에서의 활용:
   - Loss function은 대부분 scalar
   - Weight matrix W에 대한 gradient 계산
   - Backpropagation = chain rule with matrix gradients
"""

print(summary)

print("\n모든 예제 완료! 🎉")
print("이제 행렬 미분이 명확해졌나요?")
print("="*70)
