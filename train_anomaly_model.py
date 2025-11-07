import warnings
import os
import sys

# Anomalib
from anomalib.engine import Engine
import albumentations as A
from albumentations.pytorch import ToTensorV2
from anomalib.deploy import ExportType
from anomalib.loggers import AnomalibTensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.fabric.utilities.exceptions import MisconfigurationException

# PyTorch
from torch.utils.data import DataLoader
import torch
from pathlib import Path


try:
    sys.path.append('./src')
    from config import Config
    from src.utils import setup_experiment
    from src.model_selector import ModelSelector  # [신규] 모델 선택 모듈
    from src.datasets import TransitorDataset
    from PIL import Image
except ImportError:
    print("오류: 'config.py' 또는 'src/utils.py' 파일을 찾을 수 없습니다.")
    exit()

warnings.filterwarnings("ignore")


class SafeModelCheckpoint(ModelCheckpoint):
    """
    ModelCheckpoint의 확장 버전으로, metrics가 없을 때 에러를 발생시키지 않습니다.
    """
    def _save_topk_checkpoint(self, trainer, monitor_candidates):
        """metrics가 없으면 에러를 발생시키지 않고 저장을 건너뜁니다."""
        try:
            super()._save_topk_checkpoint(trainer, monitor_candidates)
        except MisconfigurationException as e:
            if "could not find the monitored key" in str(e):
                # metrics가 아직 로그되지 않았으면 저장을 건너뜀
                # save_last=True가 설정되어 있으면 최신 체크포인트는 저장됨
                # 첫 번째 validation epoch에서는 metrics가 없을 수 있음
                pass
            else:
                raise


class SafeEarlyStopping(EarlyStopping):
    """
    EarlyStopping의 확장 버전으로, metrics가 없을 때 에러를 발생시키지 않습니다.
    """
    def _validate_condition_metric(self, trainer):
        """metrics가 없으면 에러를 발생시키지 않고 False를 반환합니다."""
        try:
            return super()._validate_condition_metric(trainer)
        except RuntimeError as e:
            if "which is not available" in str(e) or "not available" in str(e):
                # metrics가 아직 로그되지 않았으면 False 반환 (계속 학습)
                return False
            else:
                raise
    
    def on_train_epoch_end(self, trainer, pl_module):
        """학습 에포크 종료 시에는 early stopping을 체크하지 않습니다."""
        # validation 후에만 체크하도록 하기 위해 train epoch end에서는 체크하지 않음
        pass
    
    def on_validation_end(self, trainer, pl_module):
        """validation 후에 early stopping을 체크합니다."""
        # validation 후에 체크하도록 재정의
        if not trainer.sanity_checking:
            self._run_early_stopping_check(trainer)


class MetricsLogger(Callback):
    """
    학습 및 검증 손실과 메트릭을 로그하는 콜백입니다.
    """
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
    
    def _get_metric_value(self, metrics_dict, key):
        """메트릭 딕셔너리에서 값을 안전하게 가져옵니다."""
        if key not in metrics_dict:
            return None
        value = metrics_dict[key]
        # Tensor인 경우 item() 호출
        if hasattr(value, 'item'):
            return value.item()
        return float(value)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """학습 에포크 종료 시 메트릭 로그"""
        # logged_metrics와 callback_metrics 모두 확인
        logged_metrics = trainer.logged_metrics if hasattr(trainer, 'logged_metrics') else {}
        callback_metrics = trainer.callback_metrics
        
        epoch = trainer.current_epoch
        
        # 학습 메트릭 수집 (여러 소스에서 확인)
        train_metrics = {}
        for key in ['train_loss', 'train/loss', 'loss']:
            value = self._get_metric_value(logged_metrics, key) or self._get_metric_value(callback_metrics, key)
            if value is not None:
                train_metrics['train_loss'] = value
                break
        
        if train_metrics:
            self.logger.info(
                f"Epoch {epoch} - Train: {', '.join([f'{k}={v:.4f}' for k, v in train_metrics.items()])}"
            )
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """검증 에포크 종료 시 메트릭 로그"""
        # logged_metrics와 callback_metrics 모두 확인
        logged_metrics = trainer.logged_metrics if hasattr(trainer, 'logged_metrics') else {}
        callback_metrics = trainer.callback_metrics
        
        epoch = trainer.current_epoch
        
        # 검증 메트릭 수집
        val_metrics = {}
        
        # val_loss 확인
        for key in ['val_loss', 'val/loss', 'validation_loss']:
            value = self._get_metric_value(logged_metrics, key) or self._get_metric_value(callback_metrics, key)
            if value is not None:
                val_metrics['val_loss'] = value
                break
        
        # image_AUROC 확인
        for key in ['image_AUROC', 'val_image_AUROC', 'image_auroc']:
            value = self._get_metric_value(logged_metrics, key) or self._get_metric_value(callback_metrics, key)
            if value is not None:
                val_metrics['image_AUROC'] = value
                break
        
        # pixel_AUROC 확인
        for key in ['pixel_AUROC', 'val_pixel_AUROC', 'pixel_auroc']:
            value = self._get_metric_value(logged_metrics, key) or self._get_metric_value(callback_metrics, key)
            if value is not None:
                val_metrics['pixel_AUROC'] = value
                break
        
        if val_metrics:
            self.logger.info(
                f"Epoch {epoch} - Validation: {', '.join([f'{k}={v:.4f}' for k, v in val_metrics.items()])}"
            )


def train_anomaly_model():
    """
    [수정됨] config의 동적 경로에 체크포인트를 저장합니다.
    """

    # 1. 설정 클래스 초기화 및 실험 설정 (로거/디렉토리)
    # [중요] Config() 객체를 생성하는 즉시 EXP_DIR 경로가 확정됩니다.
    config = Config()

    # [신규] utils의 설정 함수를 호출하여 로거와 폴더를 생성합니다.
    logger = setup_experiment(config)

    # [신규] TensorBoard 로거 설정
    version = str(config.LOG_DIR).split(os.sep)[-1]
    tensorboard_logger = AnomalibTensorBoardLogger(
        config.LOG_DIR,
        version=version,
        name='tensorboard'
        )

    try:
        IMAGE_SIZE = config.IMAGE_SIZE
    except AttributeError:
        logger.warning(f"'config.py'에 'IMAGE_SIZE'가 없습니다. (256, 256)을 사용합니다.")
        IMAGE_SIZE = (256, 256)

    # 데이터 디렉토리 설정 (TransitorDataset 사용)
    # transitor 데이터셋 경로 확인
    data_root = Path('data/transitor')  # 기본 경로
    if (config.data_dir.parent / 'transitor').exists():
        data_root = config.data_dir.parent / 'transitor'
    elif Path('data/transitor').exists():
        data_root = Path('data/transitor')
    elif config.data_dir.exists() and 'transitor' in str(config.data_dir):
        data_root = config.data_dir
    
    if not data_root.exists():
        logger.critical(f"오류: 데이터 폴더를 찾을 수 없습니다: {data_root}")
        logger.critical(f"확인된 경로: {config.data_dir}")
        return

    logger.info(f"--- 1단계 이상 탐지 모델 학습 시작 ---")
    logger.info(f"  [실험 경로]: {config.EXP_DIR}")
    logger.info(f"  [모델]: {config.ANOMALY_MODEL_NAME}")
    logger.info(f"  [데이터]: {data_root}")

    # 2. Augmentations 설정 (Albumentations - PIL Image를 numpy array로 변환)
    transform = A.Compose(
        [
            A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2(),  # PIL Image를 tensor로 변환
        ]
    )
    
    # Albumentations transform wrapper for PIL Images
    def albumentations_transform(image):
        """PIL Image를 numpy array로 변환하여 albumentations 적용"""
        import numpy as np
        if isinstance(image, Image.Image):
            image = np.array(image)
        return transform(image=image)['image']

    # [신규] 모델 선택기 초기화
    model_selector = ModelSelector()
    
    # 모델 요구사항 확인
    model_config = model_selector.get_model_config(config.ANOMALY_MODEL_NAME)
    training_config = model_selector.get_training_config(config.ANOMALY_MODEL_NAME)
    
    logger.info(f"모델 요구사항: validation={model_config['requires_validation']}, "
                f"mask={model_config['requires_mask']}")
    
    # 3. PyTorch Dataset 및 DataLoader 설정
    logger.info(f"데이터 로딩 중... (이미지 크기: {IMAGE_SIZE})")
    try:
        # Train dataset 생성
        train_dataset = TransitorDataset(
            root_dir=str(data_root),
            transform=albumentations_transform,
            mode='train'
        )
        logger.info(f"Train dataset 크기: {len(train_dataset)}")
        
        # Train dataloader 생성
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
        )

        for images, labels in train_dataloader:
            logger.info(f"샘플 배치 크기: {images.size()}")  # 배치 크기 출력
            break  # 첫 배치만 확인
        
        # Validation dataset 및 dataloader 설정
        val_dataloader = None
        
        if model_config['requires_validation']:
            # Test dataset을 validation으로 사용하거나 train에서 split
            test_dataset = TransitorDataset(
                root_dir=str(data_root),
                transform=albumentations_transform,
                mode='test'
            )
            logger.info(f"Test dataset 크기: {len(test_dataset)}")
            
            # Validation dataloader 생성
            val_dataloader = DataLoader(
                test_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY,
            )
            
            logger.info(f"Validation 데이터로더가 설정되었습니다.")
        
        logger.info(f"데이터 로딩 완료: train batches={len(train_dataloader)}")
        if val_dataloader:
            logger.info(f"Validation batches={len(val_dataloader)}")

    except Exception as e:
        logger.critical(f"데이터 로딩 실패: {e}")
        import traceback
        logger.critical(traceback.format_exc())
        return

    # 4. 모델 초기화 (모델 선택기 사용)
    logger.info(f"{config.ANOMALY_MODEL_NAME} 모델을 생성합니다.")
    model = model_selector.get_model(config.ANOMALY_MODEL_NAME)
    
    # 모델 인스턴스를 사용하여 최종 확인
    final_model_config = model_selector.get_model_config(config.ANOMALY_MODEL_NAME, model)

    # 5. 엔진(Trainer) 설정
    # [신규] TensorBoard 로거 추가
    # [수정] 모델 요구사항에 따라 동적으로 설정
    logger.info("학습 엔진(Trainer)을 설정합니다.")
    
    # 모델이 validation을 요구하는지에 따라 checkpoint callback 설정
    # [수정] training_config에서 monitor_metric 가져오기
    training_config = model_selector.get_training_config(config.ANOMALY_MODEL_NAME, model)
    
    # [신규] EarlyStopping 및 Callback 설정
    callbacks_list = []
    
    # [신규] Metrics Logger 추가 - train/val loss 표시
    metrics_logger = MetricsLogger(logger)
    callbacks_list.append(metrics_logger)
    
    if final_model_config['requires_validation']:
        # Anomalib 모델은 image_AUROC를 로그로 기록합니다
        # [수정] metrics가 로그되기 전까지는 monitor 없이 저장하고, 
        # save_last=True로 항상 최신 체크포인트를 저장합니다
        monitor_metric = training_config.get('monitor_metric', 'image_AUROC')
        checkpoint_callback = SafeModelCheckpoint(
            dirpath=config.CHECKPOINT_DIR,
            filename=f"anomalib-{{epoch:02d}}-{{{monitor_metric}:.2f}}",
            save_top_k=1,
            monitor=monitor_metric,  # metrics가 로그되면 자동으로 사용됨
            mode="max",  # AUROC는 높을수록 좋으므로 "max"
            save_last=True,  # [수정] metrics가 없어도 항상 최신 체크포인트 저장
            save_on_train_epoch_end=False,  # [수정] validation 후에만 저장
            enable_version_counter=False,  # 버전 카운터 비활성화
        )
        callbacks_list.append(checkpoint_callback)
        
        # [신규] EarlyStopping 추가 - validation metric 모니터링
        # SafeEarlyStopping을 사용하여 metrics가 없을 때 에러를 방지
        early_stopping = SafeEarlyStopping(
            monitor=monitor_metric,
            mode="max",  # AUROC는 높을수록 좋음
            patience=10,  # 10 epochs 동안 개선이 없으면 중단
            min_delta=0.001,  # 최소 개선량
            verbose=True,  # 로그 출력
        )
        callbacks_list.append(early_stopping)
        
        check_val_every_n_epoch = 1
        logger.info(f"EarlyStopping 설정: {monitor_metric} 모니터링, patience=10")
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.CHECKPOINT_DIR,
            filename="anomalib-{epoch:02d}",
            save_top_k=1,
            monitor=None,  # validation이 없으면 monitor도 None
            mode="max",
        )
        callbacks_list.append(checkpoint_callback)
        check_val_every_n_epoch = None
    
    # [수정] Engine 설정 - progress bar 및 로깅 개선
    engine = Engine(
        accelerator="auto",
        devices=1,
        max_epochs=1000,
        log_every_n_steps=1,  # 매 step마다 로그 (loss 표시를 위해)
        callbacks=callbacks_list,
        check_val_every_n_epoch=check_val_every_n_epoch,
        min_epochs=config.NUM_EPOCHS - 1,
        default_root_dir=str(config.CHECKPOINT_DIR),
        enable_checkpointing=True,
        logger=tensorboard_logger,
        detect_anomaly=True,
        enable_progress_bar=True,  # [신규] progress bar 활성화
        enable_model_summary=True,  # [신규] 모델 요약 출력
    )

    # 6. 모델 학습 시작
    # [수정] PyTorch DataLoader 사용
    logger.info("--- 학습을 시작합니다 ---")
    
    if final_model_config['requires_validation']:
        # Validation이 필요한 모델
        if val_dataloader is not None:
            logger.info("Validation 데이터로더가 설정되었습니다. 모델 성능이 에포크마다 평가됩니다.")
            engine.fit(
                model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )
        else:
            logger.warning("Validation이 필요하지만 데이터로더가 없습니다. 학습을 계속합니다.")
            engine.fit(model=model, train_dataloaders=train_dataloader)
    else:
        # Validation이 필요 없는 모델 (예: PatchCore)
        logger.info("Validation이 필요 없는 모델입니다. 학습을 진행합니다.")
        engine.fit(model=model, train_dataloaders=train_dataloader)

    logger.info("--- 학습이 완료되었습니다 ---")

    # 7. 모델 파일 정리
    last_ckpt_path_v1 = config.CHECKPOINT_DIR / "last.ckpt"
    last_ckpt_path_v2 = config.CHECKPOINT_DIR / f"{config.ANOMALY_MODEL_PATH.stem}-last.ckpt"
    final_model_path = config.ANOMALY_MODEL_PATH

    engine.export(model=model,
                  export_type=ExportType.TORCH,
                  export_root=last_ckpt_path_v2
                  )

    try:
        if last_ckpt_path_v2.exists():
            logger.info(f"최종 모델 파일 정리: {last_ckpt_path_v2.name} -> {final_model_path.name}")
            last_ckpt_path_v2.rename(final_model_path)
        elif last_ckpt_path_v1.exists():
            logger.info(f"최종 모델 파일 정리: {last_ckpt_path_v1.name} -> {final_model_path.name}")
            last_ckpt_path_v1.rename(final_model_path)
        else:
            logger.error(f"오류: 학습된 모델 파일('last.ckpt' 또는 '{last_ckpt_path_v2.name}')을 찾을 수 없습니다.")
            return

        logger.info(f"✅ 모델이 성공적으로 {final_model_path} 에 저장되었습니다.")

    except Exception as e:
        logger.error(f"모델 파일 이름 변경 중 오류 발생: {e}")


if __name__ == "__main__":
    train_anomaly_model()