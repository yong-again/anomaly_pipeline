"""
학습용 데이터 모듈
데이터셋 로딩, 분할, 데이터로더 생성을 처리합니다.
"""

import torch
from torch.utils.data import DataLoader, random_split
import albumentations as A
from typing import Optional, Tuple
import logging
from dataclasses import dataclass

from src.datasets import TransitorDatasets


@dataclass
class AnomalyBatch:
    """
    Anomalib가 기대하는 배치 형식.
    PreProcessor가 batch.image, batch.gt_mask에 접근할 수 있도록 합니다.
    update 메서드를 추가하여 Anomalib의 test_step에서 사용할 수 있도록 합니다.
    """
    image: torch.Tensor
    label: Optional[torch.Tensor] = None
    gt_mask: Optional[torch.Tensor] = None
    path: Optional[list] = None
    
    def update(self, **kwargs):
        """
        Anomalib의 test_step에서 batch.update(**predictions._asdict())를 호출할 때 사용됩니다.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # 새로운 속성 추가
                setattr(self, key, value)


def create_transforms(image_size: Tuple[int, int]) -> A.Compose:
    """
    학습용 데이터 증강 변환을 생성합니다.
    
    Args:
        image_size: 이미지 리사이징을 위한 (높이, 너비) 튜플
        
    Returns:
        Albumentations Compose 변환 객체
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        A.ToTensorV2()
    ])


def _anomaly_collate_fn(batch):
    """
    Anomalib가 기대하는 AnomalyBatch 형식으로 배치를 변환합니다.
    
    Args:
        batch: 데이터셋에서 반환된 튜플 리스트 [(image, label), ...]
        
    Returns:
        AnomalyBatch 객체
    """
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    
    # AnomalyBatch 형식으로 변환
    # PatchCore는 정상 이미지만 사용하므로 gt_mask는 None
    return AnomalyBatch(
        image=images,
        label=labels,
        gt_mask=None,  # PatchCore는 마스크가 필요 없음
        path=[f"image_{i}" for i in range(len(batch))]  # 경로 정보 (선택적)
    )


def create_dataloaders(
    data_dir,
    transform: A.Compose,
    batch_size: int,
    num_workers: int,
    requires_validation: bool,
    val_split_ratio: float = 0.3,
    random_seed: int = 42,
    logger: Optional[logging.Logger] = None
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    커스텀 데이터셋으로부터 학습 및 검증 데이터로더를 생성합니다.
    
    Args:
        data_dir: train.csv를 포함하는 데이터 디렉토리 경로
        transform: 적용할 Albumentations 변환
        batch_size: 데이터로더의 배치 크기
        num_workers: 데이터로더의 워커 수
        requires_validation: train/val로 분할할지 여부
        val_split_ratio: 검증 분할 비율 (기본값: 0.3)
        random_seed: 재현성을 위한 랜덤 시드
        logger: 로깅 메시지를 위한 선택적 로거
        
    Returns:
        (train_dataloader, val_dataloader) 튜플
        requires_validation이 False인 경우 val_dataloader는 None
    """
    if logger:
        logger.info(f"데이터 로딩 중... (이미지 크기: {transform.transforms[0].height if hasattr(transform.transforms[0], 'height') else 'N/A'})")
    
    try:
        # 전체 train 데이터셋 로드
        full_dataset = TransitorDatasets(data_dir=data_dir, mode='train', transform=transform)
        
        # 모델 요구사항에 따라 train/validation 분할
        if requires_validation:
            # Validation이 필요한 경우: train 70%, validation 30%
            train_size = int((1 - val_split_ratio) * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            train_dataset, val_dataset = random_split(
                full_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(random_seed)
            )
            
            if logger:
                logger.info(f"데이터셋 분할: Train={len(train_dataset)}, Validation={len(val_dataset)}")
            
            # 학습 데이터로더 (AnomalyBatch 형식으로 변환)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=_anomaly_collate_fn
            )
            
            # 검증 데이터로더 (AnomalyBatch 형식으로 변환)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=_anomaly_collate_fn
            )
            
            return train_dataloader, val_dataloader
        else:
            # Validation이 필요 없는 경우: 전체를 train으로 사용
            if logger:
                logger.info(f"Validation이 필요 없는 모델입니다. 전체 데이터를 train으로 사용: {len(full_dataset)}")
            
            train_dataloader = DataLoader(
                full_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=_anomaly_collate_fn  # AnomalyBatch 형식으로 변환
            )
            
            return train_dataloader, None

    except Exception as e:
        if logger:
            logger.critical(f"데이터셋 설정 실패: {e}")
            import traceback
            logger.critical(traceback.format_exc())
        raise

