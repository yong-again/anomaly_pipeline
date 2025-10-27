import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# [중요] 'src' 디렉토리에서 부모 디렉토리(프로젝트 루트)의 'config.py'를
# 임포트하기 위해 sys.path를 설정합니다.
# ---------------------------------------------------------------------------
try:
    # 현재 파일(prompt_generator.py)의 부모(src)의 부모(프로젝트 루트)
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    sys.path.append(str(PROJECT_ROOT))
    from config import Config
except ImportError:
    # 이 스크립트를 직접 실행(테스트)할 때를 대비한 예외 처리
    print("경고: 'config.py'를 임포트할 수 없습니다.")
    print("테스트 용도로만 계속 진행합니다.")
    pass  # 테스트 코드에서 Config 클래스를 찾지 못할 경우를 대비

# ---------------------------------------------------------------------------
# 'src' 내부 모듈 임포트
# ---------------------------------------------------------------------------
try:
    # 'utils.py'에서 로거 함수 임포트
    from utils import get_logger
except ImportError:
    print("오류: 'src/utils.py'를 찾을 수 없습니다.")
    print("'prompt_generator.py'와 'utils.py'가 동일한 'src' 디렉토리에 있는지 확인하세요.")
    sys.exit(1)

# 모듈 레벨 로거 초기화
logger = get_logger("PromptGenerator")


def generate_prompts_from_heatmap(
        heatmap: np.ndarray,
        threshold: float
) -> List[List[int]]:
    """
    1단계에서 생성된 히트맵(Heatmap)을 분석하여
    2단계 SAM 2에 사용할 Bounding Box 프롬프트 리스트를 생성합니다.

    Args:
        heatmap (np.ndarray): 0~1 사이로 정규화된 (H, W) 크기의 히트맵
        threshold (float): 결함 후보로 간주할 임계값 (e.g., 0.75)

    Returns:
        List[List[int]]: Bounding Box 좌표 리스트 (e.g., [[x1, y1, x2, y2], ...])
    """

    # print -> logger.info (verbose 인자 제거됨)
    logger.info(f"프롬프트 자동 생성 중... (임계값: {threshold})")

    # 1. 히트맵을 8비트 이미지로 변환 (0-255)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # 2. 임계값 적용하여 이진화 (Thresholding)
    try:
        _, binary_map = cv2.threshold(
            heatmap_uint8,
            int(threshold * 255),
            255,
            cv2.THRESH_BINARY
        )
    except Exception as e:
        # print -> logger.error
        logger.error(f"OpenCV 임계값 처리 중 오류 발생: {e}")
        logger.error(f"히트맵 최대값: {heatmap.max()}, 최소값: {heatmap.min()}")
        return []

    # 3. 이진화 맵에서 윤곽선(Contour) 찾기
    contours, _ = cv2.findContours(
        binary_map,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    prompts_boxes = []
    if not contours:
        # print -> logger.info (경고가 아닌 정보성 메시지)
        logger.info("정보: 히트맵에서 임계값을 넘는 결함 후보 영역을 찾지 못했습니다.")
        return prompts_boxes

    # 4. 각 윤곽선을 감싸는 Bounding Box 생성
    for contour in contours:
        # [선택적] 노이즈 제거
        # if cv2.contourArea(contour) < 10:
        #     continue

        x, y, w, h = cv2.boundingRect(contour)
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        prompts_boxes.append([x1, y1, x2, y2])

    # print -> logger.info
    logger.info(f"프롬프트 생성 완료: {len(prompts_boxes)}개의 BBox를 찾았습니다.")

    return prompts_boxes


if __name__ == "__main__":

    # 테스트용 로거 가져오기
    test_logger = get_logger("PromptGeneratorTest")
    test_logger.info("--- PromptGenerator 모듈 테스트 시작 ---")

    # 1. 테스트용 Config 객체 로드
    try:
        config = Config()
        test_threshold = config.HEATMAP_THRESHOLD
        test_logger.info(f"'{PROJECT_ROOT / 'config.py'}' 로드 성공.")
    except Exception as e:
        test_logger.warning(f"'config.py' 로드 실패. 기본값으로 테스트합니다. ({e})")
        test_threshold = 0.75

    # 2. 가짜 히트맵(Fake Heatmap) 생성
    test_heatmap = np.zeros((200, 200), dtype=np.float32)
    test_heatmap[30:70, 50:100] = 0.9  # 감지되어야 함
    test_heatmap[150:160, 150:160] = 0.5  # 무시되어야 함

    test_logger.info(f"테스트 히트맵 생성 (결함 2개, 임계값 {test_threshold} 이상 1개)")

    # 3. 프롬프트 생성 함수 실행 (verbose 인자 제거됨)
    generated_boxes = generate_prompts_from_heatmap(
        test_heatmap,
        test_threshold
    )

    # 4. 결과 검증
    test_logger.info(f"생성된 BBox 리스트: {generated_boxes}")

    expected_boxes = [[50, 30, 100, 70]]

    if generated_boxes == expected_boxes:
        test_logger.info("✅ 테스트 성공. 예상 BBox와 일치합니다.")
    else:
        test_logger.error("❌ 테스트 실패.")
        test_logger.error(f"  [예상값]: {expected_boxes}")
        test_logger.error(f"  [실제값]: {generated_boxes}")