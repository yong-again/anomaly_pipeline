import warnings
import os
import sys

# Anomalib
from anomalib.data import Folder
from anomalib.data.utils import ValSplitMode
from anomalib.engine import Engine
import albumentations as A
from anomalib.deploy import ExportType
from anomalib.loggers import AnomalibTensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.fabric.utilities.exceptions import MisconfigurationException


try:
    sys.path.append('./src')
    from config import Config
    from src.utils import setup_experiment
    from src.model_selector import ModelSelector  # [신규] 모델 선택 모듈
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

    # [신규] 모델 선택기 초기화
    model_selector = ModelSelector()
    
    # 모델 요구사항 확인
    model_config = model_selector.get_model_config(config.ANOMALY_MODEL_NAME)
    training_config = model_selector.get_training_config(config.ANOMALY_MODEL_NAME)
    
    logger.info(f"모델 요구사항: validation={model_config['requires_validation']}, "
                f"mask={model_config['requires_mask']}")
    
    # 3. 데이터 모듈 설정
    logger.info(f"데이터 로딩 중... (이미지 크기: {IMAGE_SIZE})")
    try:
        # 모델이 validation을 요구하는지에 따라 설정
        val_split_mode = ValSplitMode.FROM_TEST if model_config['requires_validation'] else ValSplitMode.NONE
        val_split_ratio = 0.3 if model_config['requires_validation'] else 0.0
        
        # 모델이 mask를 요구하는지에 따라 설정
        mask_dir = config.MASK_DATA_DIR if (model_config['requires_mask'] and hasattr(config, 'MASK_DATA_DIR')) else None
        
        datamodule = Folder(
            name=config.ANOMALY_MODEL_NAME,
            root=str(normal_data_root),
            normal_dir=config.NORMAL_DATA_DIR,
            abnormal_dir=config.ANOMALY_DATA_DIR,
            mask_dir=mask_dir,
            train_batch_size=config.BATCH_SIZE,
            eval_batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=config.RANDOM_SEED,
        )

        # setup() 호출
        datamodule.setup()

    except Exception as e:
        logger.critical(f"데이터 모듈 설정 실패: {e}")
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
        check_val_every_n_epoch = 1
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.CHECKPOINT_DIR,
            filename="anomalib-{epoch:02d}",
            save_top_k=1,
            monitor=None,  # validation이 없으면 monitor도 None
            mode="max",
        )
        check_val_every_n_epoch = None
    
    engine = Engine(
        accelerator="auto",
        devices=1,
        max_epochs=-1,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=check_val_every_n_epoch,
        min_epochs=config.NUM_EPOCHS - 1,
        default_root_dir=str(config.CHECKPOINT_DIR),
        enable_checkpointing=True,
        logger=tensorboard_logger,
        detect_anomaly=True
    )

    # 6. 모델 학습 시작
    # [수정] model selector를 사용하여 학습 모드 결정
    logger.info("--- 학습을 시작합니다 ---")
    
    if final_model_config['requires_validation']:
        # Validation이 필요한 모델
        val_dataloader = datamodule.val_dataloader()
        if val_dataloader is not None:
            logger.info("Validation 데이터로더가 설정되었습니다. 모델 성능이 에포크마다 평가됩니다.")
            engine.fit(model=model, datamodule=datamodule)
        else:
            logger.warning("Validation이 필요하지만 데이터로더가 없습니다. 학습을 계속합니다.")
            engine.fit(model=model, datamodule=datamodule)
    else:
        # Validation이 필요 없는 모델 (예: PatchCore)
        logger.info("Validation이 필요 없는 모델입니다. 학습을 진행합니다.")
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