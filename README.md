# Anomaly Detection and Segmentation Pipeline

This repository contains a hybrid defect detection and segmentation pipeline that combines a state-of-the-art anomaly detection model (e.g., PatchCore from Anomalib) with the Segment Anything Model 2 (SAM 2) for precise defect segmentation. The pipeline consists of two main stages: initial defect candidate detection and subsequent fine segmentation using SAM 2.


## Pipeline Overview

```angular2html
[기존 SOTA 모델] → [이상치 점수 맵 생성] → [프롬프트 자동 생성] → [SAM 2] → [정밀 결함 마스크]
```

## Directory Structure
```angular2html
hybrid_defect_detector/
├── data/
│   ├── normal_images/        # 1. Anomalib 학습을 위한 정상 이미지
│   │   ├── 000.png
│   │   └── ...
│   └── test_images/          # 2. 추론(테스트)을 위한 이미지
│       └── defect_01.png
│
├── models/                   # 3. 학습된 모델 가중치 저장
│   ├── anomalib/             #    (PatchCore 등)
│   └── sam2/                 #    (SAM 2 체크포인트)
│
├── outputs/                  # 4. 최종 결과물 저장
│   ├── heatmap_defect_01.png
│   └── mask_defect_01.png
│
├── src/                      # 5. 핵심 소스 코드
│   ├── __init__.py
│   ├── config.py             #    (설정 파일)
│   ├── anomaly_detector.py   #    (1단계: 결함 후보 탐지)
│   ├── prompt_generator.py   #    (1->2단계: 프롬프트 자동 생성)
│   ├── segmenter.py          #    (2단계: 정밀 분할)
│   └── utils.py              #    (시각화 등 보조 도구)
│
├── train_anomaly_model.py    # 6. 1단계 모델(Anomalib) 학습 스크립트
├── run_inference.py          # 7. 전체 파이프라인 실행 스크립트
└── requirements.txt          # 8. 필요 라이브러리
```

## Getting Started

### Requirements
- Python 3.11+
- anomalib
- segment-anything

### Installation
1. Clone the repository:

3. ```bash
   git clone https://github.com/yong-again/anomaly_pipeline.git
   ```
   
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
   
4. Install additional dependencies for Anomalib and SAM 2 as needed.
    ```bash
    git clone https://github.com/facebookresearch/sam2.git && cd sam2
    
    pip install -e .
    ```
   
5. Download pre-trained weights for Anomalib (e.g., PatchCore) and SAM 2, and place them in the `models/` directory.
    ```bash
   mkdir -p models/anomalib models/sam2
   s
    cd checkpoints && \
    ./download_ckpts.sh && \
   
   cd ..
   
   ```
   