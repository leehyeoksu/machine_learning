#!/usr/bin/env python3
"""
Script to generate comprehensive Jupyter notebooks for Days 4-7
"""

import json

# Day 4: Linear Algebra Core I
day4_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# 4일차: 선형대수 핵심 I - 역행렬, 행렬식, 랭크\n\n## 학습 목표\n- 역행렬(inverse) 계산과 개념 이해\n- 행렬식(determinant) 이해\n- 행렬의 랭크(rank) 이해\n- 선형 방정식 시스템 풀이"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["import numpy as np\nprint(f'NumPy 버전: {np.__version__}')"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 💡 이론 개요\n\n### 역행렬 (Inverse Matrix)\n정방행렬 A에 대해 A⁻¹는 A·A⁻¹ = I (단위행렬) 관계를 만족하는 행렬\n\n**용도**: 선형방정식 Ax=b의 해를 x=A⁻¹b로 구함\n\n**조건**: det(A) ≠ 0 (비특이행렬)\n\n### 행렬식 (Determinant)\n정방행렬에 대해 정의되는 스칼라 값\n\n**의미**: \n- 기하학적: 변환의 부피 스케일\n- 대수적: det(A)=0이면 역행렬 없음\n\n### 랭크 (Rank)\n행렬의 선형독립인 행(또는 열)의 최대 개수\n\n**의미**: 행렬이 표현할 수 있는 벡터 공간의 차원"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},"source": ["## 📚 1. 역행렬 계산\n\n### 1.1 2×2 행렬의 역행렬"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# 2×2 행렬의 역행렬\nA = np.array([[4, 7],\n              [2, 6]])\n\nprint('행렬 A:')\nprint(A)\nprint()\n\n# NumPy로 역행렬 계산\nA_inv = np.linalg.inv(A)\nprint('역행렬 A⁻¹:')\nprint(A_inv)\nprint()\n\n# 수동 계산 (2×2 공식)\n# A⁻¹ = 1/det(A) * [[d, -b], [-c, a]]\ndet_A = 4*6 - 7*2\nprint(f'행렬식: det(A) = 4×6 - 7×2 = {det_A}')\nprint()\n\nA_inv_manual = (1/det_A) * np.array([[6, -7], [-2, 4]])\nprint('수동 계산한 역행렬:')\nprint(A_inv_manual)\nprint()\n\nprint(f'NumPy 결과와 일치: {np.allclose(A_inv, A_inv_manual)}')"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# 역행렬 검증: A @ A⁻¹ = I\nidentity = A @ A_inv\nprint('A @ A⁻¹:')\nprint(identity)\nprint()\n\n# 단위행렬과 비교\nI = np.eye(2)\nprint('단위행렬 I:')\nprint(I)\nprint()\n\nprint(f'단위행렬과 일치 (허용오차 고려): {np.allclose(identity, I)}')"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 1.2 3×3 행렬의 역행렬"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# 3×3 행렬\nB = np.array([[1, 2, 3],\n              [0, 1, 4],\n              [5, 6, 0]])\n\nprint('행렬 B:')\nprint(B)\nprint()\n\n# 역행렬 계산\nB_inv = np.linalg.inv(B)\nprint('역행렬 B⁻¹:')\nprint(B_inv)\nprint()\n\n# 검증\nidentity_3 = B @ B_inv\nprint('B @ B⁻¹:')\nprint(identity_3)\nprint()\n\n# 실수 오차 확인\nprint(f'단위행렬과 일치 (허용오차): {np.allclose(identity_3, np.eye(3))}')\nprint(f'최대 오차: {np.max(np.abs(identity_3 - np.eye(3))):.2e}')"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 📚 2. 행렬식 (Determinant)\n\n### 2.1 행렬식 계산"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# 2×2 행렬의 행렬식\nA = np.array([[3, 8],\n              [4, 6]])\n\ndet_A = np.linalg.det(A)\nprint('행렬 A:')\nprint(A)\nprint()\n\nprint(f'det(A) = {det_A}')\nprint()\n\n# 수동 계산\ndet_manual = 3*6 - 8*4\nprint(f'수동 계산: 3×6 - 8×4 = {det_manual}')\nprint(f'일치: {np.isclose(det_A, det_manual)}')"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# 3×3 행렬의 행렬식\nC = np.array([[6, 1, 1],\n              [4, -2, 5],\n              [2, 8, 7]])\n\ndet_C = np.linalg.det(C)\nprint('행렬 C:')\nprint(C)\nprint(f'\\ndet(C) = {det_C:.2f}')"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 2.2 특이행렬 (Singular Matrix)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# 행렬식이 0인 특이행렬 (역행렬 없음)\nS = np.array([[1, 2, 3],\n              [2, 4, 6],\n              [3, 6, 9]])\n\nprint('특이행렬 S (두 번째 행 = 첫 번째 행 × 2):')\nprint(S)\nprint()\n\ndet_S = np.linalg.det(S)\nprint(f'det(S) = {det_S:.2e}  (0에 매우 가까움)')\nprint()\n\n# 역행렬 시도\ntry:\n    S_inv = np.linalg.inv(S)\n    print('역행렬 (불안정):', S_inv)\nexcept np.linalg.LinAlgError as e:\n    print(f'❌ 오류: {e}')\n    print('설명: det(S)=0이므로 역행렬이 존재하지 않습니다')"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 📚 3. 행렬의 랭크 (Rank)\n\n### 3.1 랭크 계산"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# 풀랭크 행렬\nA = np.array([[1, 0, 0],\n              [0, 1, 0],\n              [0, 0, 1]])\n\nrank_A = np.linalg.matrix_rank(A)\nprint('풀랭크 행렬 A (단위행렬):')\nprint(A)\nprint(f'랭크: {rank_A} (= min(3, 3) = 3) ✅')\nprint()"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# 랭크가 낮은 행렬\nB = np.array([[1, 2, 3],\n              [2, 4, 6],\n              [3, 6, 9]])\n\nrank_B = np.linalg.matrix_rank(B)\nprint('랭크 부족 행렬 B (모든 행이 선형종속):')\nprint(B)\nprint(f'랭크: {rank_B}')\nprint('설명: 세 행이 모두 [1,2,3]의 배수이므로 랭크 = 1')\nprint()"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# 직사각형 행렬의 랭크\nC = np.array([[1, 2, 3, 4],\n              [5, 6, 7, 8],\n              [9, 10, 11, 12]])\n\nrank_C = np.linalg.matrix_rank(C)\nprint('직사각형 행렬 C (3×4):')\nprint(C)\nprint(f'랭크: {rank_C}')\nprint(f'최대 가능 랭크: min(3, 4) = 3')\nprint(f'실제 랭크: {rank_C} (선형종속성 존재)')"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 📚 4. 선형 방정식 풀이\n\n### 4.1 np.linalg.solve 사용"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# 연립방정식 Ax = b 풀이\n# 2x + 3y = 8\n# x - 4y = -2\n\nA = np.array([[2, 3],\n              [1, -4]])\nb = np.array([8, -2])\n\nprint('연립방정식:')\nprint('2x + 3y = 8')\nprint('x - 4y = -2')\nprint()\n\n# 방법 1: np.linalg.solve\nx = np.linalg.solve(A, b)\nprint('해 (np.linalg.solve):')\nprint(f'x = {x[0]:.2f}, y = {x[1]:.2f}')\nprint()\n\n# 검증\nresult = A @ x\nprint('검증 (Ax):')\nprint(result)\nprint(f'b와 일치: {np.allclose(result, b)}')\nprint()"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# 방법 2: 역행렬 사용\nA_inv = np.linalg.inv(A)\nx2 = A_inv @ b\n\nprint('해 (역행렬 사용):')\nprint(f'x = {x2[0]:.2f}, y = {x2[1]:.2f}')\nprint()\n\nprint(f'두 방법의 결과 일치: {np.allclose(x, x2)}')\nprint()\nprint('※ 실무에서는 np.linalg.solve가 더 효율적이고 안정적입니다')"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 4.2 과소/과대 결정 시스템"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# 과대 결정 시스템 (방정식 개수 > 미지수 개수)\n# 해가 없거나 최소제곱해 필요\n\nA_over = np.array([[1, 1],\n                   [1, 2],\n                   [1, 3]])\nb_over = np.array([2, 3, 4])\n\nprint('과대 결정 시스템 (3개 방정식, 2개 미지수):')\nprint('A:')\nprint(A_over)\nprint(f'b: {b_over}')\nprint()\n\n# np.linalg.lstsq로 최소제곱해\nx_lstsq, residuals, rank, s = np.linalg.lstsq(A_over, b_over, rcond=None)\nprint('최소제곱해:')\nprint(x_lstsq)\nprint(f'\\n잔차 (residual): {residuals}')"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 🔥 연습 문제\n\n### 문제 1: 역행렬 계산 및 검증"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# TODO: 다음 행렬의 역행렬을 구하고 A @ A⁻¹ = I 임을 확인하세요\nA = np.array([[3, 1],\n              [5, 2]])\n\n# 여기에 코드 작성\n"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# 해답\nA = np.array([[3, 1],\n              [5, 2]])\n\nprint('행렬 A:')\nprint(A)\nprint()\n\n# 역행렬 계산\nA_inv = np.linalg.inv(A)\nprint('역행렬 A⁻¹:')\nprint(A_inv)\nprint()\n\n# 검증\nidentity = A @ A_inv\nprint('A @ A⁻¹:')\nprint(identity)\nprint()\n\nprint(f'단위행렬과 일치: {np.allclose(identity, np.eye(2))}')\n\n# 행렬식 확인\ndet_A = np.linalg.det(A)\nprint(f'\\ndet(A) = {det_A:.2f} (≠ 0이므로 역행렬 존재 ✅)')"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 문제 2: 랭크 확인"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# TODO: 다음 행렬들의 랭크를 구하고 설명하세요\n\nB1 = np.array([[1, 2, 3],\n               [2, 4, 6],\n               [3, 6, 9]])\n\nB2 = np.array([[1, 0, 0],\n               [0, 1, 1],\n               [0, 1, 1]])\n\nB3 = np.array([[1, 2],\n               [3, 4],\n               [5, 6]])\n\n# 여기에 코드 작성\n"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# 해답\nB1 = np.array([[1, 2, 3],\n               [2, 4, 6],\n               [3, 6, 9]])\nB2 = np.array([ [1, 0, 0],\n               [0, 1, 1],\n               [0, 1, 1]])\nB3 = np.array([[1, 2],\n               [3, 4],\n               [5, 6]])\n\nprint('B1 (모든 행이 선형종속):')\nprint(B1)\nprint(f'랭크: {np.linalg.matrix_rank(B1)}')\nprint('설명: 모든 행이 [1,2,3]의 배수\\n')\n\nprint('B2 (두 번째와 세 번째 행이 동일):')\nprint(B2)\nprint(f'랭크: {np.linalg.matrix_rank(B2)}')\nprint('설명: 선형독립인 행이 2개\\n')\n\nprint('B3 (직사각형, 풀랭크):')\nprint(B3)\nprint(f'랭크: {np.linalg.matrix_rank(B3)}')\nprint(f'최대 가능 랭크: min(3, 2) = 2')\nprint('설명: 두 열이 모두 선형독립')"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 📝 복습 및 팁\n\n### 핵심 정리\n\n| 개념 | 의미 | NumPy 함수 | 조건 |\n|------|------|-----------|------|\n| 역행렬 | A·A⁻¹ = I | `np.linalg.inv()` | det(A) ≠ 0 |\n| 행렬식 | 부피 스케일 | `np.linalg.det()` | 정방행렬 |\n| 랭크 | 선형독립 차원 | `np.linalg.matrix_rank()` | 모든 행렬 |\n| 선형방정식 | Ax = b | `np.linalg.solve()` | A가 정방/비특이 |\n\n### 주요 개념\n1. **가역행렬**: det(A) ≠ 0, 풀랭크\n2. **특이행렬**: det(A) = 0, 역행렬 없음\n3. **풀랭크**: rank = min(행, 열)\n\n### 딥러닝 연관성\n- 역행렬: 이론적 이해 (실제로는 경사하강 사용)\n- 행렬식: Hessian 분석\n- 랭크: 모델 표현력 분석"]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write Day 4
with open('/home/hyuksu/projects/ml/study2/Day4_linear_algebra_core1.ipynb', 'w') as f:
    json.dump(day4_content, f, indent=1, ensure_ascii=False)

print("Day 4 notebook created successfully!")
