import sys
import cv2
import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# [중요] 'src' 디렉토리에서 부모 디렉토리(프로젝트 루트)의 'config.py'를
# 임포트하기 위해 sys.path를 설정합니다.
# ---------------------------------------------------------------------------
try:
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    sys.path.append(str(PROJECT_ROOT))
    from config import Config
except ImportError:
    print("오류: 'config.py'를 찾을 수 없습니다.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# [신규] 실험 경로 및 로거 설정 함수
# ---------------------------------------------------------------------------

# 로거가 중복 설정되는 것을 방지하기 위한 플래그
_logger_initialized = False


def setup_experiment(config: Config) -> logging.Logger:
    """
    [신규] 메인 스크립트에서 호출할 단일 설정 함수.
    1. config의 make_dirs()를 호출하여 모든 실험 폴더를 생성합니다.
    2. 로거를 설정하여 콘솔과 'config.LOG_DIR'에 로그 파일을 생성합니다.
    """
    global _logger_initialized, _log_dir

    # 1. 모든 실험 디렉토리 생성
    try:
        config.make_dirs()
    except Exception as e:
        print(f"디렉토리 생성 실패: {e}")
        # 계속 진행하되, 로그 파일 저장은 실패할 수 있음

    # 2. 로거 설정 (기존 get_logger 로직 통합)
    logger = logging.getLogger("HybridDetector")  # 루트 로거 이름
    logger.setLevel(logging.INFO)  # 항상 INFO 레벨로 설정

    if _logger_initialized or logger.hasHandlers():
        return logger  # 이미 설정됨

    # 콘솔 핸들러
    console_level = logging.INFO if config.VERBOSE else logging.WARNING
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (config.LOG_DIR 사용)
    log_file_path = config.LOG_DIR / "pipeline.log"
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    _logger_initialized = True

    logger.info(f"로거 초기화 완료. 로그 파일: {log_file_path}")
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    [수정] 설정된 루트 로거의 자식 로거를 가져옵니다.
    이 함수는 'setup_experiment'가 호출된 *이후*에 사용되어야 합니다.
    """
    return logging.getLogger(f"HybridDetector.{name}")


# ---------------------------------------------------------------------------
# 2. 시각화 (Visualization)
# ---------------------------------------------------------------------------

def visualize_results(
        config: Config,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        sam_mask: np.ndarray,
        prompts_boxes: List[List[int]],
        filename: str
) -> Path:
    """
    [수정] 결과 저장 경로가 'config.VISUALIZATION_DIR'로 변경되었습니다.
    """
    # 로거 가져오기 (setup_experiment가 이미 호출되었다고 가정)
    logger = get_logger("Visualizer")

    # --- 1. 원본 + BBox (Panel 1) ---
    vis_original = original_image.copy()
    for (x1, y1, x2, y2) in prompts_boxes:
        cv2.rectangle(vis_original, (x1, y1), (x2, y2), (0, 255, 0), 2)
    _add_label(vis_original, "Original + Prompts")

    # --- 2. 히트맵 (Panel 2) ---
    vis_heatmap = (heatmap * 255).astype(np.uint8)
    vis_heatmap_color = cv2.applyColorMap(vis_heatmap, cv2.COLORMAP_JET)
    _add_label(vis_heatmap_color, "1. Anomaly Heatmap")

    # --- 3. SAM 마스크 오버레이 (Panel 3) ---
    vis_sam_overlay = original_image.copy()
    red_mask = np.zeros_like(vis_sam_overlay)
    red_mask[sam_mask] = (0, 0, 255)  # (B, G, R)
    vis_sam_overlay = cv2.addWeighted(vis_sam_overlay, 0.7, red_mask, 0.3, 0)
    for (x1, y1, x2, y2) in prompts_boxes:
        cv2.rectangle(vis_sam_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    _add_label(vis_sam_overlay, "2. SAM Segmentation")

    # --- 4. 이미지 병합 및 저장 ---
    try:
        final_image = np.hstack([vis_original, vis_heatmap_color, vis_sam_overlay])

        # [수정] 저장 경로가 'VISUALIZATION_DIR'로 변경됨
        save_path = config.VISUALIZATION_DIR / f"result_{filename}"

        cv2.imwrite(str(save_path), final_image)
        logger.info(f"시각화 결과 저장 완료: {save_path}")

        return save_path

    except Exception as e:
        logger.error(f"시각화 이미지 병합 또는 저장 실패: {e}")
        return None


def _add_label(image: np.ndarray, text: str):
    """이미지 좌측 상단에 텍스트 레이블을 추가하는 헬퍼 함수"""
    cv2.putText(
        image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (255, 255, 255), 2, cv2.LINE_AA
    )


# ---------------------------------------------------------------------------
# [테스트]
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    print("--- Utils 모듈 테스트 시작 ---")

    config = Config()

    # [수정] setup_experiment를 호출하여 로거와 디렉토리 초기화
    logger = setup_experiment(config)

    logger.info("[1. 로거 테스트] 로거 및 실험 경로 설정 완료.")
    logger.info(f"실험 경로: {config.EXP_DIR}")

    # --- 2. 시각화 테스트 ---
    logger.info("[2. 시각화 테스트]")
    try:
        h, w = 256, 256
        test_image = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(test_image, (w // 2, h // 2), 50, (255, 100, 0), -1)
        test_heatmap = np.zeros((h, w), dtype=np.float32)
        cv2.circle(test_heatmap, (w // 2, h // 2 - 10), 60, 0.9, -1)
        test_mask = np.zeros((h, w), dtype=bool)
        cv2.circle(test_mask, (w // 2, h // 2), 55, True, -1)
        test_boxes = [[w // 2 - 60, h // 2 - 70, w // 2 + 60, h // 2 + 50]]
        test_filename = "utils_test.png"

        save_path = visualize_results(
            config, test_image, test_heatmap, test_mask, test_boxes, test_filename
        )

        if save_path and save_path.exists():
            logger.info(f"✅ 시각화 테스트 성공. 결과가 {save_path} 에 저장되었습니다.")
        else:
            logger.error("❌ 시각화 테스트 실패.")

    except Exception as e:
        logger.error(f"❌ 시각화 테스트 중 오류 발생: {e}")

    print("--- Utils 모듈 테스트 종료 ---")