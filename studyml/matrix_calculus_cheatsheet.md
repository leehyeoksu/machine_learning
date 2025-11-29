# 행렬 미분 치트시트 (Cheat Sheet)
## Matrix Calculus Quick Reference

---

## 🎯 핵심 원칙

### Rule #1: Shape Matching
```
∂f/∂A의 shape = A의 shape (항상!)
```

### Rule #2: Layout Convention
```
ML에서는 Denominator Layout 사용
→ ∂(Ax)/∂x = A^T (전치!)
```

### Rule #3: Element-wise
```
(∂f/∂A)ᵢⱼ = ∂f/∂Aᵢⱼ
```

---

## 📐 벡터 미분 공식

| Function | Gradient | Notes |
|----------|----------|-------|
| `a^T x` | `a` | 선형 |
| `x^T a` | `a` | 동일 |
| `x^T x` | `2x` | Quadratic |
| `Ax` | `A^T` | **전치 주의!** |
| `x^T A` | `A` | 전치 없음 |
| `x^T Ax` | `(A + A^T)x` | A 대칭이면 `2Ax` |
| `‖x‖₂` | `x/‖x‖₂` | Normalization |
| `‖x‖₂²` | `2x` | MSE |
| `‖Ax - b‖²` | `2A^T(Ax - b)` | Least Squares |

---

## 🔲 행렬 미분 공식

### Trace 기본 공식

| Function | Gradient | Condition |
|----------|----------|-----------|
| `tr(A)` | `I` | |
| `tr(AB)` | `B^T` | B 고정 |
| `tr(A^T B)` | `B` | B 고정 |
| `tr(ABA^T)` | `AB^T + AB` | B 대칭이면 `2AB` |
| `tr(A^T BA)` | `(B + B^T)A` | B 대칭이면 `2BA` |
| `tr(A^k)` | `k(A^(k-1))^T` | |

### Norm 공식

| Function | Gradient |
|----------|----------|
| `‖A‖_F²` | `2A` |
| `‖AX‖_F²` | `2AXX^T` |
| `‖AX - B‖_F²` | `2(AX - B)X^T` |
| `‖XA‖_F²` | `2X^TXA` |

### 행렬식 (Determinant)

| Function | Gradient |
|----------|----------|
| `\|A\|` | `\|A\|(A^(-1))^T` |
| `log\|A\|` | `(A^(-1))^T` |

---

## 🧮 자주 쓰는 패턴

### Linear Regression
```
L(w) = ‖Xw - y‖²

∂L/∂w = 2X^T(Xw - y)

최적해: w* = (X^T X)^(-1) X^T y
```

### Ridge Regression
```
L(w) = ‖Xw - y‖² + λ‖w‖²

∂L/∂w = 2X^T(Xw - y) + 2λw

최적해: w* = (X^T X + λI)^(-1) X^T y
```

### Weight Matrix Gradient (Neural Networks)
```
L = ‖WX - Y‖²

∂L/∂W = 2(WX - Y)X^T
```

### Logistic Regression
```
L(w) = -Σ[y log σ(w^T x) + (1-y)log(1-σ(w^T x))]

∂L/∂w = X^T(σ(Xw) - y)
```

### Softmax + Cross-Entropy
```
L = -Σ y_i log σ(z)_i

∂L/∂z = σ(z) - y  ← 매우 간단!
```

---

## 🔗 Chain Rule

### 벡터 Chain Rule
```
z = f(y), y = g(x)

∂z/∂x = (∂y/∂x)^T · (∂z/∂y)
```

**예제:**
```
L = ‖Ax - b‖²
u = Ax - b
L = u^T u

∂L/∂x = (∂u/∂x)^T · (∂L/∂u)
      = A^T · (2u)
      = 2A^T(Ax - b)
```

### 행렬 Chain Rule
```
L = tr(f(A))

∂L/∂A = (∂f/∂A)^T · (∂L/∂f)  (in trace form)
```

---

## 🎨 Trace Tricks

### Cyclic Property
```
tr(ABC) = tr(BCA) = tr(CAB)
```

### Transpose Invariance
```
tr(A^T) = tr(A)
```

### Scalar to Trace
```
x^T Ax = tr(x^T Ax) = tr(Axx^T)
```

### 활용 예시
```
L = ‖WX - Y‖_F²
  = tr((WX - Y)^T(WX - Y))
  = tr(X^T W^T WX - 2X^T W^T Y + Y^T Y)

∂L/∂W = 2(WX - Y)X^T
```

---

## 📝 Jacobian Matrix

### 정의
```
y = f(x), y ∈ R^m, x ∈ R^n

J = ∂y/∂x ∈ R^(m×n)

Jᵢⱼ = ∂yᵢ/∂xⱼ
```

### 예제
```
y = Ax, A ∈ R^(m×n)

J = A ∈ R^(m×n)

하지만 scalar loss의 경우:
∂L/∂x = A^T · (∂L/∂y)  ← 전치!
```

---

## 🔍 검증 방법

### Numerical Gradient
```python
def numerical_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2*eps)
    return grad
```

### Shape Check Workflow
```
1. Input shape 확인
2. Output shape 예상
3. Gradient shape = Input shape 확인
4. Numerical gradient로 검증
```

---

## ⚠️ 자주 하는 실수

### ❌ 실수 1: 전치 빠뜨림
```
❌ ∂(Ax)/∂x = A
✅ ∂(Ax)/∂x = A^T
```

### ❌ 실수 2: Chain Rule에서 차원 안 맞춤
```
❌ ∂L/∂x = ∂L/∂y · ∂y/∂x
✅ ∂L/∂x = (∂y/∂x)^T · (∂L/∂y)
```

### ❌ 실수 3: Scalar vs Vector 헷갈림
```
L = x^T Ax (scalar)

❌ ∂L/∂x = Ax
✅ ∂L/∂x = (A + A^T)x
```

### ❌ 실수 4: Layout Convention 혼동
```
Numerator layout과 Denominator layout이 다름!
ML에서는 Denominator layout 사용
```

---

## 💡 빠른 유도 전략

### Strategy 1: 간단한 예제로 시작
```
2×2 행렬로 element별로 계산
→ 패턴 발견
→ 일반화
```

### Strategy 2: Trace 활용
```
Scalar를 trace로 변환
→ Trace 미분 공식 사용
→ Cyclic property로 정리
```

### Strategy 3: Element-wise 접근
```
(∂f/∂A)ᵢⱼ = ∂f/∂Aᵢⱼ

특정 element에 대해 편미분
→ 전체 행렬로 조립
```

---

## 📚 차원별 분류

### Scalar → Vector
```
f: R → R^n
∂f/∂x ∈ R^n
```

### Vector → Scalar
```
f: R^n → R
∂f/∂x ∈ R^n (gradient)
```

### Vector → Vector
```
f: R^n → R^m
∂f/∂x ∈ R^(m×n) (Jacobian)
```

### Matrix → Scalar (가장 중요!)
```
f: R^(m×n) → R
∂f/∂A ∈ R^(m×n) (gradient)
```

### Matrix → Matrix (거의 안 씀)
```
f: R^(m×n) → R^(p×q)
∂f/∂A ∈ R^(pq×mn) (Jacobian, 4D tensor)
```

---

## 🚀 실전 팁

### Tip 1: Shape First!
모든 계산 전에 shape 먼저 확인

### Tip 2: 공식 외우지 말고 유도
기본 공식 몇 개만 기억하고 나머지는 유도

### Tip 3: Numerical로 항상 검증
Analytical gradient 구한 후 반드시 numerical로 체크

### Tip 4: Vectorization
For loop 대신 행렬 연산 사용

### Tip 5: Dimension Matching
Chain rule 쓸 때 차원 항상 확인

---

## 🎯 암기해야 할 최소 공식

**이것만 외우면 된다:**

```
1. ∂(Ax)/∂x = A^T
2. ∂(x^T Ax)/∂x = (A + A^T)x
3. ∂tr(AB)/∂A = B^T
4. ∂‖Ax - b‖²/∂x = 2A^T(Ax - b)
```

나머지는 위 4개로 유도 가능!

---

## 📖 참고 자료

- Matrix Cookbook (Petersen & Pedersen)
- CS231n Backpropagation Notes
- Deep Learning Book (Goodfellow et al.) - Chapter 2
- ML_L04_vector.calculus_review.pdf

---

**이 치트시트를 저장해두고 필요할 때마다 참고하세요!** 📌
