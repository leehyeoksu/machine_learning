import json
import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler

class WineClassifierHandler(BaseHandler):
    """Wine 데이터 분류를 위한 Custom Handler"""
    
    def preprocess(self, data):
        """입력 데이터 전처리 - JSON에서 tensor로 변환"""
        inputs = []
        for row in data:
            # JSON body에서 데이터 추출
            if isinstance(row, dict):
                body = row.get("body") or row.get("data")
            else:
                body = row
            
            if isinstance(body, (bytes, bytearray)):
                body = body.decode('utf-8')
            
            if isinstance(body, str):
                body = json.loads(body)
            
            # features 리스트를 tensor로 변환
            if isinstance(body, dict):
                features = body.get("features") or body.get("data")
            else:
                features = body
            
            inputs.append(features)
        
        # Tensor로 변환
        input_tensor = torch.FloatTensor(inputs).to(self.device)
        return input_tensor
    
    def inference(self, data):
        """모델 추론 실행"""
        with torch.no_grad():
            logits = self.model(data)
            # Softmax로 확률 변환
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def postprocess(self, inference_output):
        """추론 결과 후처리"""
        results = []
        for output in inference_output:
            # 최대 확률 클래스와 확률값
            prob, pred_class = torch.max(output, 0)
            result = {
                "predicted_class": pred_class.item(),
                "probability": prob.item(),
                "all_probabilities": output.tolist()
            }
            results.append(result)
        return results
