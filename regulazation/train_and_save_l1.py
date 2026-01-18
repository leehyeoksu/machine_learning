import torch
import torch.nn as nn
import torch.optim as optim
from dataset_wine import get_wine_loaders, set_seed
from model import MLP

set_seed(42)
device = torch.device('cpu')

# 데이터 로더 준비
train_loader, val_loader, test_loader, input_dim, num_classes = get_wine_loaders(batch_size=64)

# L1 정규화 모델 학습
print("=" * 60)
print("L1 정규화 모델 학습 중...")
print("=" * 60)

model_l1 = MLP(input_dim, num_classes).to(device)

def train_l1(model, train_loader, val_loader, epochs=50, lr=1e-3, l1_lambda=1e-4):
    opt = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    best = {'val_acc': 0.0, 'state': None}
    
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            ce_loss = ce(logits, y)
            
            # L1 정규화 계산
            l1 = 0.0
            for name, p in model.named_parameters():
                if p.requires_grad and p.dim() > 1:
                    l1 = l1 + p.abs().sum()
            
            loss = ce_loss + l1_lambda * l1
            loss.backward()
            opt.step()
            running += loss.item()
        
        # Validation accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        
        if val_acc > best['val_acc']:
            best['val_acc'] = val_acc
            best['state'] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        if ep % 10 == 0:
            print(f'Epoch {ep:02d} | loss={running/len(train_loader):.4f} | val_acc={val_acc:.4f}')
    
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    return model, best['val_acc']

model_l1, val_acc = train_l1(model_l1, train_loader, val_loader, epochs=50)

# 모델 저장
torch.save(model_l1.state_dict(), 'wine_l1.pth')
print(f"\n✓ L1 모델 저장 완료: wine_l1.pth (val_acc={val_acc:.4f})")
