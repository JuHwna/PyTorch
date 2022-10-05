# 실습 환경 설정과 파이토치 기초
## 파이토치 기초 문법
### 텐서 다루기
- 텐서 생성
 
```python
import torch
print(torch.tensor([[1,2],[3,4]])) # 2차원 형태의 텐서 생성
print(torch.tensor([[1,2],[3,4]]),device="cuda:0")) # GPU에 텐서 생성
print(torch.tensor([[1,2],[3,4]],dtype=torch.float64)) # dtype을 이용하여 텐서 생성
```

- 텐서를 ndarray로 변환
```python
torch.tensor([[1,2],[3,4]]).numpy() # 텐서를 ndarray로 변환
torch.tensor([[1,2],[3,4]],device="cuda:0") # GPU 상의 텐서를 CPU의 텐서로 변환한 후 ndarray로 변환
```

### 텐서의 자료형
- torch.FloatTensor : 32비트의 부동 소수점
- torch.DoubleTensor : 64비트의 부동 소수점
- torch.LongTensor : 64비트의 부호가 있는 정수

### 텐서의 차원 조작(변경)
- view 사용
  - numpy의 reshape과 유사함

```python
temp=torch.tensor([[1,2],[3,4,]]) # 2x2 행렬 생성
print(temp.view(4,1)) # 2x2 행렬을 4x1로 변형
print(temp.view(-1)) # 2x2 행렬을 1차원 벡터로 변형
print(temp.view(1,-1)) # 2x2 행렬을 1x4로 변형
print(temp.view(-1,1)) # 2x2 행렬을 4x1로 변형
```

- cat : 다른 길이의 텐서를 하나로 병합할 때 사용
- transpose : 행렬의 전치 외에도 차원의 순서를 변경할 때도 사용됨

### 데이터 준비
