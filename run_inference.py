import sys
import argparse
from pathlib import Path
import logging  # [신규]

# --- 1. 필수 모듈 임포트 ---
try:
    from config import Config
except ImportError:
    print("치명적 오류: 'config.py' 파일을 찾을 수 없습니다.")
    sys.exit(1)

SRC_DIR = Path(__file__).parent / "src"
sys.path.append(str(SRC_DIR))

try:
    # [수정] setup_experiment 함수 임포트
    from utils import setup_experiment, visualize_results, get_logger
    from anomaly_detector import AnomalyDetector
    from prompt_generator import generate_prompts_from_heatmap
    from segmenter import Segmenter
except ImportError as e:
    print(f"치명적 오류: 'src' 디렉토리에서 모듈을 임포트하지 못했습니다: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------

def run_hybrid_pipeline(
        image_path: Path,
        config: Config,
        logger: "logging.Logger",
        detector: AnomalyDetector,
        segmenter: Segmenter
):
    logger.info(f"--- 파이프라인 시작: {image_path.name} ---")

    # --- 0. 이미지 로드 ---
    import cv2
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"이미지 로드 실패: {image_path}")
        return

    logger.info(f"이미지 로드 완료 (Shape: {image.shape})")

    # --- 1. 1단계: 결함 후보 영역 탐지 (Anomalib) ---
    try:
        heatmap = detector.predict(image)
        logger.info("1단계: Anomalib 히트맵 생성 완료.")
    except Exception as e:
        logger.error(f"1단계 (AnomalyDetector) 실행 중 오류: {e}")
        return

    # --- 2. 프롬프트 자동 생성 ---
    try:
        prompts_boxes = generate_prompts_from_heatmap(
            heatmap,
            config.HEATMAP_THRESHOLD
        )
        logger.info(f"프롬프트 생성 완료. {len(prompts_boxes)}개의 BBox 감지.")
    except Exception as e:
        logger.error(f"프롬프트 생성 (generate_prompts_from_heatmap) 중 오류: {e}")
        return

    # --- 3. 2단계: 결함 정밀 분할 (SAM) ---
    try:
        final_mask = segmenter.segment(image, prompts_boxes)
        logger.info("2단계: SAM 정밀 마스크 생성 완료.")
    except Exception as e:
        logger.error(f"2단계 (Segmenter) 실행 중 오류: {e}")
        return

    # --- 4. 결과 시각화 및 저장 ---
    try:
        visualize_results(
            config=config,
            original_image=image,
            heatmap=heatmap,
            sam_mask=final_mask,
            prompts_boxes=prompts_boxes,
            filename=image_path.name
        )
    except Exception as e:
        logger.error(f"시각화 (visualize_results) 중 오류: {e}")
        return

    logger.info(f"--- 파이프라인 종료: {image_path.name} ---")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="2-Stage Hybrid Anomaly Detection Pipeline")
    parser.add_argument("-i", "--input", type=str, required=True)
    args = parser.parse_args()

    # --- 2. 설정 및 로거 초기화 (중요) ---
    try:
        # [수정] Config 객체를 *한 번만* 생성합니다.
        config = Config()

        # [수정] setup_experiment를 호출하여 로거와 디렉토리를 설정합니다.
        logger = setup_experiment(config)

    except Exception as e:
        print(f"치명적 오류: 설정(Config) 또는 로거 초기화 실패: {e}")
        sys.exit(1)

    logger.info(f"설정 로드 완료. (Device: {config.DEVICE})")
    logger.info(f"실험 경로: {config.EXP_DIR}")

    # --- 3. 모델 사전 로드 (중요) ---
    logger.info("=" * 50)
    logger.info("파이프라인 모델을 사전 로드합니다...")
    try:
        # [수정] 생성자에 config 객체를 전달해야 합니다.
        # (이를 위해 AnomalyDetector와 Segmenter의 __init__ 수정 필요)

        # --- (필수 수정) ---
        # anomaly_detector.py와 segmenter.py의 __init__을
        # def __init__(self): -> def __init__(self, config: Config):
        # self.config = Config() -> self.config = config
        # 로 수정해야 합니다.

        detector = AnomalyDetector(config)
        segmenter = Segmenter(config)

        logger.info("✅ 1단계(Anomalib) 및 2단계(SAM) 모델 로드 성공.")
    except Exception as e:
        logger.critical(f"❌ 모델 로드 실패: {e}")
        logger.critical("AnomalyDetector/Segmenter의 __init__이 config 객체를 받도록 수정했는지 확인하세요.")
        sys.exit(1)
    logger.info("=" * 50)

    # --- 4. 입력 경로 처리 ---
    input_path = Path(args.input)
    # ... (이하 로직 동일) ...

    # (생략) ...
    # ... 입력 경로 처리 ...
    image_paths_to_process = []
    if input_path.is_dir():
        logger.info(f"디렉토리에서 이미지 파일 검색 중: {input_path}")
        supported_extensions = [".png", ".jpg", ".jpeg", ".bmp"]
        for ext in supported_extensions:
            image_paths_to_process.extend(list(input_path.glob(f"*{ext}")))
    elif input_path.is_file():
        image_paths_to_process = [input_path]

    # --- 5. 각 이미지에 대해 파이프라인 실행 ---
    if not image_paths_to_process:
        logger.warning(f"처리할 이미지를 찾을 수 없습니다: {input_path}")
    else:
        logger.info(f"총 {len(image_paths_to_process)}개의 이미지 처리를 시작합니다.")

        for image_path in image_paths_to_process:
            try:
                run_hybrid_pipeline(
                    image_path=image_path,
                    config=config,  # [유지] 동일한 config 객체 전달
                    logger=logger,
                    detector=detector,
                    segmenter=segmenter
                )
            except Exception as e:
                logger.error(f"!!! {image_path.name} 처리 중 예측하지 못한 오류 발생: {e} !!!")

        logger.info("=" * 50)
        logger.info("모든 이미지 처리가 완료되었습니다.")
        logger.info(f"결과 저장 위치: {config.VISUALIZATION_DIR}")
        logger.info(f"로그 파일 위치: {config.LOG_DIR}")
        logger.info("=" * 50)