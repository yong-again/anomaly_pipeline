"""
Anomalib 모델 테스트 스크립트
학습된 PatchCore 모델을 사용하여 테스트 데이터셋에 대한 추론 및 평가를 수행합니다.
"""

from typing import Any
import warnings
import os
import sys
from pathlib import Path
import torch
import json

# Anomalib 라이브러리
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.loggers import AnomalibTensorBoardLogger

# 경로 설정
from test_config import TestConfig
sys.path.append('./src')

try:
    from src.data_module import (
        create_transforms,
        create_dataloaders,
        AnomalyBatch
    )
    from torch.utils.data import DataLoader
except ImportError as e:
    print(f"오류: 모듈을 import할 수 없습니다: {e}")
    print("확인 사항:")
    print("  1. 'test_config.py' 파일이 루트 디렉토리에 있는지 확인")
    print("  2. 'src/data_module.py' 파일이 존재하는지 확인")
    import traceback
    traceback.print_exc()
    exit()

warnings.filterwarnings("ignore")


def _test_collate_fn(batch):
    """
    테스트용 collate 함수.
    테스트 모드에서는 이미지만 반환하므로 label이 없습니다.
    AnomalyBatch를 반환하여 PreProcessor가 batch.image에 접근할 수 있도록 합니다.
    """
    images = torch.stack([item for item in batch])
    
    # AnomalyBatch 형식으로 변환 (label은 None, update 메서드 포함)
    return AnomalyBatch(
        image=images,
        label=None,
        gt_mask=None,
        path=[f"test_image_{i}" for i in range(len(batch))]
    )


def test_anomaly_model(model_path: Path = None, exp_dir: Path = None, data_dir: Path = None):
    """
    학습된 Anomalib 모델을 사용하여 테스트 데이터셋에 대한 추론 및 평가를 수행합니다.
    
    Args:
        model_path: 모델 체크포인트 경로 (None이면 config에서 자동으로 찾음)
        exp_dir: 기존 실험 디렉토리 경로 (예: results/2025-11-07/exp1) (필수)
        data_dir: 테스트 데이터 디렉토리 경로 (None이면 기본값 사용)
    """
    from src.datasets import TransitorDatasets
    
    # 1. 필수 파라미터 검증
    if exp_dir is None:
        print("오류: exp_dir이 필수입니다. --exp_dir 인자를 제공해주세요.")
        return
    
    exp_dir = Path(exp_dir)
    if not exp_dir.exists():
        print(f"오류: 지정된 실험 디렉토리가 존재하지 않습니다: {exp_dir}")
        return
    
    # data_dir 설정 (기본값: data/transitor)
    if data_dir is None:
        data_dir = Path.cwd() / "data" / "transitor"
    else:
        data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"오류: 데이터 디렉토리가 존재하지 않습니다: {data_dir}")
        return
    
    # 2. TestConfig 초기화 (새로운 디렉토리 생성하지 않음)
    try:
        config = TestConfig(exp_dir=exp_dir, data_dir=data_dir)
    except ValueError as e:
        print(f"오류: {e}")
        return
    
    # 3. 로거 설정 (setup_experiment는 사용하지 않음 - 디렉토리 생성 방지)
    import logging
    logger = logging.getLogger("TestAnomalyModel")
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (기존 로그 디렉토리 사용)
    if config.LOG_DIR.exists():
        log_file_path = config.LOG_DIR / "test_pipeline.log"
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    
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
    
    logger.info(f"--- PatchCore 모델 테스트 시작 ---")
    logger.info(f"  [실험 경로]: {config.EXP_DIR}")
    logger.info(f"  [모델]: {config.ANOMALY_MODEL_NAME}")
    logger.info(f"  [백본]: {config.BACKBONE}")
    logger.info(f"  [테스트 데이터]: {config.data_dir}")
    
    # 4. 모델 경로 확인
    if model_path is None:
        # TestConfig에서 찾은 모델 경로 사용
        if config.ANOMALY_MODEL_PATH is None or not config.ANOMALY_MODEL_PATH.exists():
            logger.critical(f"모델 파일을 찾을 수 없습니다.")
            logger.critical(f"확인 경로:")
            logger.critical(f"  {config.CHECKPOINT_DIR}")
            logger.critical(f"체크포인트 디렉토리에서 모델 파일을 찾을 수 없습니다.")
            return
        model_path = config.ANOMALY_MODEL_PATH
        logger.info(f"모델 경로: {model_path}")
    else:
        model_path = Path(model_path)
        if not model_path.exists():
            logger.critical(f"지정된 모델 경로가 존재하지 않습니다: {model_path}")
            return
    
    # 모델 정보 및 데이터셋 정보 저장
    test_info = {
        "model": {
            "name": config.ANOMALY_MODEL_NAME,
            "backbone": config.BACKBONE,
            "layers": list(config.LAYERS),
            "pre_trained": config.PRE_TRAINED,
            "coreset_sampling_ratio": config.CORESET_SAMPLING_RATIO,
            "num_neighbors": config.NUM_NEIGHBORS,
            "model_path": str(model_path)
        },
        "dataset": {
            "data_dir": str(config.data_dir),
            "image_size": list(IMAGE_SIZE) if isinstance(IMAGE_SIZE, tuple) else IMAGE_SIZE
        },
        "experiment": {
            "exp_dir": str(config.EXP_DIR),
            "test_results_dir": str(config.TEST_RESULTS_DIR)
        }
    }
    
    # test_info.json 파일로 저장
    test_info_path = config.TEST_RESULTS_DIR / "test_info.json"
    with open(test_info_path, 'w', encoding='utf-8') as f:
        json.dump(test_info, f, indent=2, ensure_ascii=False)
    logger.info(f"테스트 정보 저장 완료: {test_info_path}")
    
    # 3. 테스트 데이터셋 및 데이터로더 설정
    logger.info("테스트 데이터셋 로딩 중...")
    transform = create_transforms(IMAGE_SIZE)
    
    try:
        test_dataset = TransitorDatasets(
            data_dir=config.data_dir,
            mode='test',
            transform=transform
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=_test_collate_fn
        )
        
        logger.info(f"테스트 데이터셋 크기: {len(test_dataset)}")
    except Exception as e:
        logger.critical(f"테스트 데이터셋 설정 실패: {e}")
        import traceback
        logger.critical(traceback.format_exc())
        return
    
    # 4. 모델 로드
    logger.info(f"모델 로딩 중: {model_path}")
    try:
        # Lightning 모델의 load_from_checkpoint 사용 (권장 방법)
        try:
            model = Patchcore.load_from_checkpoint(
                str(model_path),
                backbone=config.BACKBONE,
                layers=config.LAYERS,
                pre_trained=config.PRE_TRAINED,
                coreset_sampling_ratio=config.CORESET_SAMPLING_RATIO,
                num_neighbors=config.NUM_NEIGHBORS,
                pre_processor=config.PRE_PROCESSOR,
                post_processor=config.POST_PROCESSOR,
                evaluator=config.EVALUATOR,
                visualizer=config.VISUALIZER,
                map_location=config.DEVICE
            )
            # 모델을 명시적으로 디바이스로 이동
            model = model.to(config.DEVICE)
            logger.info("Lightning load_from_checkpoint로 모델 로딩 완료")
        except Exception as e1:
            logger.warning(f"load_from_checkpoint 실패, 수동 로딩 시도: {e1}")
            # 수동 로딩 시도
            # PatchCore 모델 초기화 (학습 시와 동일한 설정)
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
            
            checkpoint = torch.load(model_path, map_location=config.DEVICE)
            
            # Lightning 모델의 경우 state_dict 키가 'model.'로 시작할 수 있음
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # 'model.' prefix 제거
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        new_state_dict[k.replace('model.', '')] = v
                    else:
                        new_state_dict[k] = v
                model.load_state_dict(new_state_dict, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            # 모델을 명시적으로 디바이스로 이동
            model = model.to(config.DEVICE)
            logger.info("수동 로딩으로 모델 로딩 완료")
        
        model.eval()
        # 모델이 올바른 디바이스에 있는지 확인
        model = model.to(config.DEVICE)
        
        # 모델의 첫 번째 파라미터가 올바른 디바이스에 있는지 확인
        next_param = next(model.parameters(), None)
        if next_param is not None:
            model_device = next_param.device
            logger.info(f"모델 디바이스: {model_device}, 요청된 디바이스: {config.DEVICE}")
            if str(model_device) != str(config.DEVICE):
                logger.warning(f"모델이 {model_device}에 있지만 {config.DEVICE}로 이동해야 합니다.")
                model = model.to(config.DEVICE)
        
        logger.info("모델 준비 완료")
        
    except Exception as e:
        logger.critical(f"모델 로딩 실패: {e}")
        import traceback
        logger.critical(traceback.format_exc())
        return
    
    # 5. 테스트 엔진 설정
    logger.info("테스트 엔진 설정 중...")
    engine = Engine(
        accelerator="auto",
        devices=1,
        logger=tensorboard_logger,
        default_root_dir=str(config.TEST_RESULTS_DIR)
    )
    
    # 6. 테스트 실행
    logger.info("--- 테스트 시작 ---")
    try:
        # 모델이 올바른 디바이스에 있는지 확인
        model = model.to(config.DEVICE)
        model.eval()
        
        # 모델의 첫 번째 파라미터가 올바른 디바이스에 있는지 확인
        next_param = next(model.parameters(), None)
        if next_param is not None:
            model_device = next_param.device
            logger.info(f"테스트 전 모델 디바이스 확인: {model_device}, 요청된 디바이스: {config.DEVICE}")
            if str(model_device) != str(config.DEVICE):
                logger.warning(f"모델이 {model_device}에 있지만 {config.DEVICE}로 이동합니다.")
                model = model.to(config.DEVICE)
        
        # Anomalib Engine의 test 메서드 사용
        test_results = engine.test(
            model=model,
            dataloaders=test_dataloader
        )
        
        logger.info("--- 테스트 완료 ---")
        logger.info(f"테스트 결과: {test_results}")
        
        # 결과 요약 출력 및 저장
        if test_results and len(test_results) > 0:
            logger.info("=" * 50)
            logger.info("테스트 결과 요약:")
            for key, value in test_results[0].items():
                logger.info(f"  {key}: {value}")
            logger.info("=" * 50)
            
            # 테스트 결과를 JSON 파일로 저장
            test_results_dict = {}
            for key, value in test_results[0].items():
                # torch.Tensor나 numpy array는 리스트로 변환
                if isinstance(value, torch.Tensor):
                    test_results_dict[key] = value.cpu().tolist() if value.numel() == 1 else value.cpu().numpy().tolist()
                elif hasattr(value, 'tolist'):  # numpy array
                    test_results_dict[key] = value.tolist()
                else:
                    test_results_dict[key] = value
            
            test_results_path = config.TEST_RESULTS_DIR / "test_results.json"
            with open(test_results_path, 'w', encoding='utf-8') as f:
                json.dump(test_results_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"테스트 결과 저장 완료: {test_results_path}")
        
    except Exception as e:
        logger.error(f"테스트 실행 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Engine.test가 실패하는 경우 수동으로 추론 수행
        logger.info("수동 추론 모드로 전환...")
        model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            # 모델이 올바른 디바이스에 있는지 확인
            model = model.to(config.DEVICE)
            
            for batch_idx, batch in enumerate(test_dataloader):
                # batch가 AnomalyBatch 형태이므로 속성 접근 방식 사용
                images = batch.image.to(config.DEVICE)
                
                # 모델과 데이터가 같은 디바이스에 있는지 확인
                next_param = next(model.parameters(), None)
                if next_param is not None and next_param.device != images.device:
                    logger.warning(f"모델 디바이스({next_param.device})와 데이터 디바이스({images.device})가 일치하지 않습니다. 모델을 {images.device}로 이동합니다.")
                    model = model.to(images.device)
                
                # 모델 추론
                outputs = model(images)
                
                # 예측 결과 저장
                if isinstance(outputs, dict):
                    anomaly_scores = outputs.get('anomaly_score', outputs.get('pred_score', None))
                    anomaly_maps = outputs.get('anomaly_map', None)
                    
                    if anomaly_scores is not None:
                        all_predictions.extend(anomaly_scores.cpu().numpy())
                    
                    logger.info(f"배치 {batch_idx + 1}/{len(test_dataloader)} 처리 완료")
                else:
                    logger.warning(f"예상치 못한 출력 형식: {type(outputs)}")
        
        logger.info(f"총 {len(all_predictions)}개 샘플 추론 완료")
    
    logger.info("✅ 테스트 완료")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Anomalib 모델 테스트")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="모델 체크포인트 경로 (지정하지 않으면 config에서 자동으로 찾음)"
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="기존 실험 디렉토리 경로 (예: results/2025-11-07/exp1) (필수)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="테스트 데이터 디렉토리 경로 (기본값: data/transitor)"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path) if args.model_path else None
    exp_dir = Path(args.exp_dir) if args.exp_dir else None
    data_dir = Path(args.data_dir) if args.data_dir else None
    test_anomaly_model(model_path=model_path, exp_dir=exp_dir, data_dir=data_dir)

