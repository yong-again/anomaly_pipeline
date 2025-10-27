import warnings
from pathlib import Path
import logging  # [신규] logging 임포트

# Anomalib
from anomalib.data import Folder
from anomalib import TaskType
from anomalib.data.utils import ValSplitMode
from anomalib.models import get_model
from anomalib.engine import Engine
import albumentations as A
from anomalib.deploy import ExportType

from albumentations.pytorch import ToTensorV2
from lightning.pytorch.callbacks import ModelCheckpoint

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
    # 이 함수는 config.make_dirs()를 내부적으로 호출합니다.
    logger = setup_experiment(config)

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
            #A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            #ToTensorV2(),
        ]
    )

    # 3. 데이터 모듈 설정
    logger.info(f"데이터 로딩 중... (이미지 크기: {IMAGE_SIZE})")
    try:
        datamodule = Folder(
            name=f"{config.ANOMALY_MODEL_NAME}_dataset",
            root=str(normal_data_root),
            normal_dir=config.NORMAL_DATA_DIR,
            train_batch_size=config.BATCH_SIZE,
            eval_batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            val_split_mode=ValSplitMode.FROM_TRAIN,
            val_split_ratio=0.15,
            seed=config.RANDOM_SEED,
        )
        datamodule.setup()
    except Exception as e:
        logger.critical(f"데이터 모듈 설정 실패: {e}")
        return


    # 4. 모델 초기화
    logger.info(f"{config.ANOMALY_MODEL_NAME} 모델을 생성합니다.")
    model = get_model(config.ANOMALY_MODEL_NAME)

    # 5. 콜백(Callback) 설정
    logger.info(f"체크포인트 저장 경로: {config.CHECKPOINT_DIR}")
    # checkpoint_callback = [
    #     ModelCheckpoint(
    #     # 동적 실험 경로 하위 'checkpoints' 폴더
    #     dirpath=str(config.CHECKPOINT_DIR),
    #     # 'anomaly_model' (config에서 정의)
    #     filename=config.ANOMALY_MODEL_PATH.stem,
    #     # 정상 데이터만 학습하므로 모니터링할 지표 없음
    #     monitor=None,
    #     # 'best' 모델 저장을 비활성화
    #     save_top_k=0,
    #     # 마지막 에포크의 모델을 저장
    #     save_last=True,
    #     )
    # ]

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=str(config.CHECKPOINT_DIR),
    #     filename='best-model-{epoch:02d}',
    #     save_top_k=1,
    #     monitor='image_AUROC',
    #     mode='max'
    # )

    # 6. 엔진(Trainer) 설정
    logger.info("학습 엔진(Trainer)을 설정합니다.")
    engine = Engine(
        accelerator="auto",
        devices=1,
        #callbacks=[checkpoint_callback],
        max_epochs=config.NUM_EPOCHS,
        default_root_dir='./outputs',
        enable_checkpointing=True,
        logger=True
    )

    # 7. 모델 학습 시작
    logger.info("--- 학습을 시작합니다 ---")
    engine.fit(model=model, datamodule=datamodule)
    logger.info("--- 학습이 완료되었습니다 ---")

    # 8. 모델 파일 정리
    # [수정] config.CHECKPOINT_DIR에서 파일을 찾습니다.
    last_ckpt_path_v1 = config.CHECKPOINT_DIR / "last.ckpt"
    last_ckpt_path_v2 = config.CHECKPOINT_DIR / f"{config.ANOMALY_MODEL_PATH.stem}-last.ckpt"
    final_model_path = config.ANOMALY_MODEL_PATH

    engine.export(model=model,
                  export_type=ExportType.TORCH,
                  export_root=final_model_path
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