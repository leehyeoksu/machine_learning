# 7일 학습 계획: NumPy와 PyTorch로 배우는 선형대수 및 딥러닝 기초

완성된 노트북 학습 가이드

## 📚 학습 커리큘럼

### ✅ Day 1: NumPy 기본과 선형대수 도입
**파일**: `Day1_numpy_basics.ipynb`
- NumPy 배열 생성 및 조작
- 벡터와 행렬 기본 개념
- 원소별 연산 vs 행렬 연산
- Shape, ndim, dtype 이해

### ✅ Day 2: 차원 조작
**파일**: `Day2_dimension_manipulation.ipynb`
- Reshape로 배열 형태 변경
- Transpose로 축 교환
- Expand_dims, Squeeze로 차원 추가/제거
- Concatenate, Stack으로 배열 합치기

### ✅ Day 3: 텐서 연산
**파일**: `Day3_tensor_operations.ipynb`
- 행렬 곱셈 (matmul) 상세 설명
- 브로드캐스팅 규칙과 응용
- 집계 연산 (sum, mean, max, min)
- 비교 연산과 필터링

### 🔄 Day 4: 선형대수 핵심 I
**파일**: `Day4_linear_algebra_core1.ipynb`
- 역행렬 계산과 검증
- 행렬식(Determinant)
- 랭크(Rank) 이해
- 선형 방정식 풀이

### 🔄 Day 5: 선형대수 핵심 II
**파일**: `Day5_linear_algebra_core2.ipynb`
- 유사역행렬 (Pseudoinverse)
- SVD (특잇값 분해)
- 고유값과 고유벡터
- QR 분해와 Matrix Norm

### 🔄 Day 6: PyTorch 텐서와 Autograd
**파일**: `Day6_pytorch_autograd.ipynb`
- PyTorch Tensor 기본
- Autograd 메커니즘
- Gradient 계산과 역전파
- Gradient 추적 제어

### 🔄 Day 7: PyTorch 모델 구성
**파일**: `Day7_pytorch_models.ipynb`
- nn.Module 클래스 구조
- 손실 함수와 옵티마이저
- 학습 루프 구현
- Dataset과 DataLoader

## 🚀 사용 방법

1. **순서대로 학습**: Day 1부터 시작하여 순차적으로 진행
2. **노트북 실행**: Jupyter Notebook 또는 VS Code에서 실행
3. **코드 실습**: 모든 셀을 직접 실행하며 결과 확인
4. **연습 문제**: 각 노트북에 포함된 연습 문제 풀이

## 💻 실행 환경

```bash
# Jupyter Notebook 실행
cd /home/hyuksu/projects/ml/study2
jupyter notebook

# 또는 특정 노트북 열기
jupyter notebook Day1_numpy_basics.ipynb
```

## 📋 필요 라이브러리

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
```

## 🎯 학습 목표

- NumPy 배열 조작 능력 향상
- 선형대수 핵심 개념 이해
- PyTorch 기본 사용법 습득
- 딥러닝 기초 준비 완료

## 📝 참고사항

- 각 노트북은 독립적으로 실행 가능
- 모든 계산 과정이 상세히 설명되어 있음
- 한국어로 작성된 이론 설명
- 실습 문제와 해답 포함

---

**학습 진행 상황**: Days 1-3 완료 ✅ | Days 4-7 진행 중 🔄
