# TorchServe Quick Start

## 🚀 복붙용 명령어 (한방에 실행)

터미널에 아래 코드를 그대로 복사해서 붙여넣으세요.

```bash
# 1. 프로젝트 폴더 이동 및 가상환경 활성화
cd /home/hyuksu/projects/ml/regulazation
source /home/hyuksu/projects/ml/.venv/bin/activate

# 2. 기존 서버 중지 (혹시 켜져있으면)
torchserve --stop 2>/dev/null

# 3. 모델 학습 및 패키징 (필요한 경우)
python train_and_save_l1.py
python train_and_save_l2.py
mkdir -p model_store
torch-model-archiver --model-name wine_l1 --version 1.0 --model-file model.py --serialized-file wine_l1.pth --handler wine_handler.py --export-path model_store --force
torch-model-archiver --model-name wine_l2 --version 1.0 --model-file model.py --serialized-file wine_l2.pth --handler wine_handler.py --export-path model_store --force

# 4. 서버 시작
torchserve --start --ts-config config.properties --model-store model_store --models wine_l1=wine_l1.mar wine_l2=wine_l2.mar --ncs

# 5. 테스트 (3초 대기 후 실행)
sleep 3
curl -X POST http://127.0.0.1:8080/predictions/wine_l1 -H "Content-Type: application/json" -d '{"features": [13.2, 2.77, 2.51, 18.5, 96.0, 1.9, 0.58, 0.63, 1.14, 7.5, 0.72, 1.88, 415.0]}'
```

## 🛑 서버 종료 명령어

```bash
torchserve --stop
```
