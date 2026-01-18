"""
MLP 모델 성능 비교: Softmax vs No-Softmax (MLflow 통합 버전)

이 스크립트는 Softmax가 있는 모델과 없는 모델의 성능을 비교하고
MLflow를 사용하여 실험을 추적합니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from data import trainloader, testloader, classes
from nonsoftmaxmlp import MLP_NoSoftmax
from softmaxmlp import MLP_WithSoftmax
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Softmax_vs_NoSoftmax")

print("데이터 로딩 완료!")


def train_and_evaluate(model, model_name, epochs=5, lr=0.001):
    """
    모델 학습 및 평가 (MLflow 통합)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    use_softmax_head = isinstance(model, MLP_WithSoftmax)
    criterion = nn.NLLLoss() if use_softmax_head else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # MLflow 파라미터 로깅
    mlflow.log_param("lr", lr)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("model_type", model_name)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("device", str(device))
    
    train_accs = []
    test_accs = []
    
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            if use_softmax_head:
                probs = outputs.clamp_min(1e-12)
                loss = criterion(torch.log(probs), labels)
                preds = probs.argmax(dim=1)
            else:
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
            
            loss.backward()
            optimizer.step()
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        
        train_acc = 100.0 * correct / total
        train_accs.append(train_acc)
        
        # Evaluation
        model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        test_acc = 100.0 * correct / total
        test_accs.append(test_acc)
        
        # MLflow 메트릭 로깅
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("test_acc", test_acc, step=epoch)
        
        print(f"{model_name} - Epoch [{epoch+1}/{epochs}] Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")
    
    # ★ 모델 저장 (HTTP 서버로 문제없이 업로드됨)
    mlflow.pytorch.log_model(model, "model")
    
    return train_accs, test_accs


if __name__ == "__main__":
    print("=" * 60)
    print("MLP (No Softmax) 학습 시작")
    print("=" * 60)
    
    # No Softmax run
    with mlflow.start_run(run_name="No-Softmax-Model"):
        model_no_softmax = MLP_NoSoftmax()
        train_no, test_no = train_and_evaluate(model_no_softmax, "No Softmax", epochs=10)
    
    print("\n" + "=" * 60)
    print("MLP (With Softmax) 학습 시작")
    print("=" * 60)
    
    # With Softmax run
    with mlflow.start_run(run_name="With-Softmax-Model"):
        model_with_softmax = MLP_WithSoftmax()
        train_with, test_with = train_and_evaluate(model_with_softmax, "With Softmax", epochs=10)
    
    # 성능 비교 그래프
    epochs = range(1, len(train_no) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_no, 'b-o', label='No Softmax')
    plt.plot(epochs, train_with, 'r-s', label='With Softmax')
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy (%)')
    plt.title('Train Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_no, 'b-o', label='No Softmax')
    plt.plot(epochs, test_with, 'r-s', label='With Softmax')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
    
    print("\n" + "=" * 60)
    print("최종 성능 비교")
    print("=" * 60)
    print(f"No Softmax  - Train: {train_no[-1]:.2f}% | Test: {test_no[-1]:.2f}%")
    print(f"With Softmax - Train: {train_with[-1]:.2f}% | Test: {test_with[-1]:.2f}%")
    
    print("\n실험 결과가 MLflow에 저장되었습니다!")
    print("MLflow UI를 확인하려면 터미널에서 'mlflow ui' 명령을 실행하세요.")
