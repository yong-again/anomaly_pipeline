import warnings
import os
import sys

# Anomalib 라이브러리
from anomalib.engine import Engine
from anomalib.loggers import AnomalibTensorBoardLogger
from anomalib.models import Patchcore

# 경로 설정
from config import Config
sys.path.append('./src')

try:
    from src.utils import setup_experiment
    from src.data_module import (
        create_transforms,
        create_dataloaders
    )
except ImportError as e:
    print(f"오류: 모듈을 import할 수 없습니다: {e}")
    print("확인 사항:")
    print("  1. 'config.py' 파일이 루트 디렉토리에 있는지 확인")
    print("  2. 'src/utils.py' 파일이 존재하는지 확인")
    import traceback
    traceback.print_exc()
    exit()

warnings.filterwarnings("ignore")


def train_anomaly_model():
    """
    [수정됨] config의 동적 경로에 체크포인트를 저장합니다.
    """

    # 1. 설정 클래스 초기화 및 실험 설정 (로거/디렉토리)
    # [중요] Config() 객체를 생성하는 즉시 EXP_DIR 경로가 확정됩니다.
    config = Config()

    # [신규] utils의 설정 함수를 호출하여 로거와 폴더를 생성합니다.
    logger = setup_experiment(config)

    # TensorBoard 로거 설정
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

    logger.info(f"--- PatchCore 이상 탐지 모델 학습 시작 ---")
    logger.info(f"  [실험 경로]: {config.EXP_DIR}")
    logger.info(f"  [모델]: {config.ANOMALY_MODEL_NAME}")
    logger.info(f"  [백본]: {config.BACKBONE}")
    logger.info(f"  [레이어]: {config.LAYERS}")
    logger.info(f"  [코어셋 샘플링 비율]: {config.CORESET_SAMPLING_RATIO}")
    logger.info(f"  [이웃 수]: {config.NUM_NEIGHBORS}")
    logger.info(f"  [데이터]: {config.data_dir}")

    # 2. 데이터셋 및 데이터로더 설정
    # PatchCore는 validation이 필요 없으므로 전체 데이터를 train으로 사용
    transform = create_transforms(IMAGE_SIZE)
    
    try:
        train_dataloader, val_dataloader = create_dataloaders(
            data_dir=config.data_dir,
            transform=transform,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            requires_validation=False,  # PatchCore는 validation 불필요
            val_split_ratio=0.3,
            random_seed=42,
            logger=logger
        )
    except Exception as e:
        logger.critical(f"데이터셋 설정 실패: {e}")
        import traceback
        logger.critical(traceback.format_exc())
        return

    # 3. PatchCore 모델 초기화
    logger.info(f"PatchCore 모델을 생성합니다. (백본: {config.BACKBONE})")
    try:
        model = Patchcore(
            backbone=config.BACKBONE,
            layers=config.LAYERS,
            pre_trained=config.PRE_TRAINED,
            coreset_sampling_ratio=config.CORESET_SAMPLING_RATIO,
            num_neighbors=config.NUM_NEIGHBORS,
            pre_processor=config.PRE_PROCESSOR,
            post_processor=config.POST_PROCESSOR,
            evaluator=config.EVALUATOR,
            visualizer=config.VISUALIZER
        )
        logger.info(f"PatchCore 모델 생성 완료: {config.BACKBONE}")
    except Exception as e:
        logger.critical(f"모델 생성 실패: {e}")
        import traceback
        logger.critical(traceback.format_exc())
        return

    # 4. 학습 엔진(Trainer) 설정
    logger.info("학습 엔진(Trainer)을 설정합니다.")
    
    engine = Engine(
        accelerator="auto",
        devices=1,
        max_epochs=-1,
        log_every_n_steps=1,
        min_epochs=config.NUM_EPOCHS - 1,
        default_root_dir=str(config.CHECKPOINT_DIR),
        enable_checkpointing=True,
        logger=tensorboard_logger,
        detect_anomaly=True
    )

    # 5. 모델 학습 시작
    logger.info("--- 학습을 시작합니다 ---")
    logger.info("PatchCore는 validation이 필요 없는 모델입니다. 학습을 진행합니다.")
    
    engine.fit(model=model, train_dataloaders=train_dataloader)

    logger.info("--- 학습이 완료되었습니다 ---")
    logger.info(f"✅ 모델 체크포인트가 {config.CHECKPOINT_DIR}에 저장되었습니다.")


if __name__ == "__main__":
    train_anomaly_model()