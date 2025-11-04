import warnings
import os

# Anomalib
from anomalib.data import Folder
from anomalib.data.utils import ValSplitMode
from anomalib.models import get_model
from anomalib.engine import Engine
import albumentations as A
from anomalib.deploy import ExportType
from anomalib.loggers import AnomalibTensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from anomalib.metrics import AUROC, F1Score

# Config 및 [신규] setup_experiment 임포트
try:
    from src.config import Config
    from src.utils import setup_experiment  # [신규]
except ImportError:
    print("오류: 'config.py' 또는 'src/utils.py' 파일을 찾을 수 없습니다.")
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

    normal_data_root = config.data_dir / "normal_dir"
    if not normal_data_root.exists():
        logger.critical(f"오류: 정상 이미지 데이터 폴더를 찾을 수 없습니다: {normal_data_root}")
        return

    logger.info(f"--- 1단계 이상 탐지 모델 학습 시작 ---")
    logger.info(f"  [실험 경로]: {config.EXP_DIR}")
    logger.info(f"  [모델]: {config.ANOMALY_MODEL_NAME}")
    logger.info(f"  [데이터]: {normal_data_root}")

    # 2. Augmentations 설정
    transform = A.Compose(
        [
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ]
    )

    # 3. 데이터 모듈 설정
    logger.info(f"데이터 로딩 중... (이미지 크기: {IMAGE_SIZE})")
    try:
        datamodule = Folder(
            name=config.ANOMALY_MODEL_NAME,
            root=str(normal_data_root),
            normal_dir=config.NORMAL_DATA_DIR,
            abnormal_dir=config.ANOMALY_DATA_DIR,
            mask_dir = config.MASK_DATA_DIR if hasattr(config, 'MASK_DATA_DIR') else None,
            train_batch_size=config.BATCH_SIZE,
            eval_batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            val_split_mode=ValSplitMode.FROM_TEST,
            val_split_ratio=0.3,
            seed=config.RANDOM_SEED,
        )

        # setup() 호출
        datamodule.setup()

    except Exception as e:
        logger.critical(f"데이터 모듈 설정 실패: {e}")
        return

    # 4. 모델 초기화
    logger.info(f"{config.ANOMALY_MODEL_NAME} 모델을 생성합니다.")
    model = get_model(config.ANOMALY_MODEL_NAME)

    # 5. 엔진(Trainer) 설정
    # [신규] TensorBoard 로거 추가
    logger.info("학습 엔진(Trainer)을 설정합니다.")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="anomalib-{epoch:02d}-{image_AUROC:.2f}",
        save_top_k=1,
        monitor='image_AUROC',
        mode="max",
    )
    engine = Engine(
        accelerator="auto",
        devices=1,
        max_epochs=-1,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
        min_epochs=config.NUM_EPOCHS - 1,
        default_root_dir=str(config.CHECKPOINT_DIR),
        enable_checkpointing=True,
        logger=tensorboard_logger,
        detect_anomaly=True
    )

    # 6. 모델 학습 시작
    logger.info("--- 학습을 시작합니다 ---")
    if datamodule.val_dataloader() is not None:
        print("Validation 데이터로더가 설정되었습니다. 모델 성능이 에포크마다 평가됩니다.")
        engine.fit(model=model, datamodule=datamodule)
    else:
        print("Validation 데이터로더가 설정되지 않았습니다. 모델 성능이 평가되지 않습니다.")
        checkpoint_callback.monitor = None
        engine.fit(model=model, datamodule=datamodule)

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