# Satellite Building Segmentation — Loss Function Comparison
# 위성 이미지 건물 세그멘테이션 — 손실 함수 비교 실험

> **Euron 9기 Pixels 팀 프로젝트** | [SW중심대학 공동 AI 경진대회 2023 (DACON, 종료)](https://dacon.io/competitions/official/236092/overview/description)

---

## Overview / 프로젝트 개요

**[EN]**
Binary semantic segmentation of buildings from satellite imagery.
We compared three loss functions (BCE, Dice, BCE+Dice) using an EfficientNet-B5 + FPN architecture to find the optimal training objective.

**[KO]**
위성 이미지에서 건물을 픽셀 단위로 분류하는 이진 세그멘테이션 프로젝트입니다.
EfficientNet-B5 + FPN 구조를 기반으로, 세 가지 손실 함수(BCE, Dice, BCE+Dice)를 동일 조건에서 비교 실험하였습니다.

---

## Model Architecture / 모델 구조

| Component | Detail |
|-----------|--------|
| Encoder | EfficientNet-B5 (ImageNet pretrained) |
| Decoder | FPN (Feature Pyramid Network) |
| Output | Binary mask (1 class) |
| Library | `segmentation-models-pytorch` |

EfficientNet-B5를 인코더로, FPN(Feature Pyramid Network)을 디코더로 사용합니다.
FPN은 다양한 해상도의 피처 맵을 결합하여 크고 작은 건물을 모두 효과적으로 탐지합니다.

<img width="2780" height="1505" alt="Model Architecture: EfficientNet-B5 + FPN" src="https://github.com/user-attachments/assets/a9c92f57-79d6-4208-a8b2-fa97bce9fb68" />

---

## Training Workflow / 학습 워크플로우

세 가지 손실 함수(BCE, Dice, BCE+Dice)를 **동일한 초기 조건**에서 독립적으로 학습시켜 공정하게 비교하였습니다.
각 실험은 동일한 모델 가중치 초기화, 동일한 Train/Val split, 동일한 하이퍼파라미터를 사용합니다.

<img width="3077" height="1617" alt="Research & Training Workflow: Loss Function Comparison" src="https://github.com/user-attachments/assets/823a9a71-d63c-46c9-81e8-cbe08003715f" />

---

## Experiment Results / 실험 결과

> 5 Epochs, AdamW (lr=1e-4, weight_decay=1e-5), Batch size=2, Input size=512×512
> Train/Val split: 80/20 (seed=11), Total: 7,140 images

| Loss Function | Best Epoch | Val IoU | Val Dice |
|---------------|-----------|---------|----------|
| BCE | 5 | 0.6945 | 0.8034 |
| Dice | 5 | 0.6822 | 0.7943 |
| **BCE + Dice** | **4** | **0.7008** | **0.8094** |

**BCE+Dice 조합이 가장 높은 성능을 기록했습니다.**

### Training Curves / 학습 곡선

Epoch별 Val IoU 추이:

| Epoch | BCE | Dice | BCE+Dice |
|-------|-----|------|----------|
| 1 | 0.6029 | 0.6345 | 0.6439 |
| 2 | 0.6514 | 0.6536 | 0.6757 |
| 3 | 0.6815 | 0.6669 | 0.6879 |
| 4 | 0.6851 | 0.6779 | **0.7008** |
| 5 | **0.6945** | **0.6822** | 0.6991 |

---

## Dataset / 데이터셋

- **Task**: Binary segmentation (건물 여부)
- **Format**: `img_id`, `img_path`, `mask_rle` (Run-Length Encoding)
- **Train set**: 7,140 images
- **Train/Val**: 5,712 / 1,428

> 데이터셋은 대회 규정상 포함하지 않습니다.

---

## Data Augmentation / 데이터 증강

```python
train_tf = A.Compose([
    A.PadIfNeeded(min_height=512, min_width=512),
    A.RandomCrop(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),        # 위성 이미지는 회전 불변
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```

---

## Environment / 실험 환경

- Python 3.12
- PyTorch (CUDA)
- Google Colab (Tesla T4, VRAM 14.74 GB, RAM 51 GB)

---

## Project Structure / 폴더 구조

```
satellite-building-segmentation/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── EfficientNet_B5_v3_loss_compare.ipynb   # 메인 실험 노트북
├── results/
│   ├── history_BCE.csv                          # BCE 학습 기록
│   ├── history_Dice.csv                         # Dice 학습 기록
│   ├── history_BCE_+_Dice.csv                   # BCE+Dice 학습 기록
│   └── final_summary_by_loss.csv               # 최종 성능 요약
└── docs/
    └── Euron_9기_Pixels.pdf                     # 발표 자료
```

---

## How to Run / 실행 방법

1. Google Colab에서 `notebooks/EfficientNet_B5_v3_loss_compare.ipynb`를 열고 Google Drive를 마운트합니다.
2. `PATHS` 딕셔너리의 경로를 본인 Drive 경로로 수정합니다.
3. 셀을 순서대로 실행합니다.

```python
# 경로 설정 예시
PATHS = {
    "train_csv": "/content/drive/MyDrive/YOUR_FOLDER/data/train.csv",
    "train_img_dir": "/content/drive/MyDrive/YOUR_FOLDER/data",
    "out_dir": "/content/drive/MyDrive/YOUR_FOLDER/data/preprocessed",
    "test_csv": "/content/drive/MyDrive/YOUR_FOLDER/data/test.csv",
    "test_img_dir": "/content/drive/MyDrive/YOUR_FOLDER/data",
}
```

---

## Team / 팀

**Euron 9기 Pixels**

---

## References / 참고 자료

- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
