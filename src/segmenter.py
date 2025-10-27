import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# [중요] 'src' 디렉토리에서 부모 디렉토리(프로젝트 루트)의 'config.py'를
# 임포트하기 위해 sys.path를 설정합니다.
# ---------------------------------------------------------------------------
try:
    # 현재 파일(segmenter.py)의 부모(src)의 부모(프로젝트 루트)
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    sys.path.append(str(PROJECT_ROOT))
    from config import Config
except ImportError:
    print("오류: 'config.py'를 찾을 수 없습니다.")
    print("프로젝트 루트 디렉토리에 'config.py' 파일이 있는지 확인하세요.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 'src' 내부 모듈 임포트
# ---------------------------------------------------------------------------
try:
    # 'utils.py'에서 로거 함수 임포트
    from utils import get_logger
except ImportError:
    print("오류: 'src/utils.py'를 찾을 수 없습니다.")
    print("'segmenter.py'와 'utils.py'가 동일한 'src' 디렉토리에 있는지 확인하세요.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# [필수] 'segment-anything' 라이브러리 임포트
# (로거를 사용하여 오류 메시지 출력)
# ---------------------------------------------------------------------------
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    # config/utils가 로드된 후이므로 로거 사용 가능
    setup_logger = get_logger("SegmenterSetup")
    setup_logger.critical("=" * 50)
    setup_logger.critical("오류: 'segment-anything' 라이브러리를 찾을 수 없습니다.")
    setup_logger.critical("SAM 모델을 사용하려면 라이브러리를 설치해야 합니다.")
    setup_logger.critical("터미널에서 다음 명령어를 실행하세요:")
    setup_logger.critical("pip install 'git+https://github.com/facebookresearch/segment-anything.git'")
    setup_logger.critical("=" * 50)
    sys.exit(1)


class Segmenter:
    """
    2단계: 결함 정밀 분할 (Precise Segmentation)

    config에 정의된 SAM 체크포인트와
    prompt_generator가 생성한 BBox 프롬프트를 사용하여
    정밀한 결함 마스크를 생성합니다.
    """

    def __init__(self):
        """
        Segmenter 초기화.
        Config 로드, SAM 모델 로드, SamPredictor 설정을 수행합니다.
        """
        self.config = Config()
        self.logger = get_logger("Segmenter")  # 로거 초기화

        if not self.config.SAM_MODEL_PATH.exists():
            # print -> logger.critical
            self.logger.critical(f"오류: 2단계 SAM 모델 파일을 찾을 수 없습니다.")
            self.logger.critical(f"경로: {self.config.SAM_MODEL_PATH}")
            self.logger.critical("config에 설정된 경로에 SAM 모델(.pth)을 다운로드하세요.")
            sys.exit(1)

        self.logger.info("2단계 모델(SAM) 로딩 중...")
        self.model_type = self._get_model_type_from_path(self.config.SAM_MODEL_PATH)

        if not self.model_type:
            # print -> logger.critical
            self.logger.critical(f"오류: {self.config.SAM_MODEL_PATH.name}에서 모델 타입을")
            self.logger.critical("('vit_h', 'vit_l', 'vit_b') 추론할 수 없습니다.")
            sys.exit(1)

        # 1. SAM 모델 아키텍처 로드
        sam_model = sam_model_registry[self.model_type](
            checkpoint=self.config.SAM_MODEL_PATH
        )

        # 2. 모델을 Config에 지정된 디바이스로 이동
        sam_model.to(device=self.config.DEVICE)

        # 3. SamPredictor 초기화
        self.predictor = SamPredictor(sam_model)

        # print -> logger.info
        self.logger.info(f"'{self.model_type}' SAM 모델 로딩 성공. (Device: {self.config.DEVICE})")

    def _get_model_type_from_path(self, model_path: Path) -> str:
        """SAM 모델 파일 이름에서 모델 타입(vit_h, vit_l, vit_b)을 추론합니다."""
        name = model_path.name.lower()
        if "vit_h" in name:
            return "vit_h"
        elif "vit_l" in name:
            return "vit_l"
        elif "vit_b" in name:
            return "vit_b"
        return ""

    def segment(self, image: np.ndarray, prompts_boxes: List[List[int]]) -> np.ndarray:
        """
        입력 이미지와 BBox 프롬프트를 받아 정밀 마스크를 반환합니다.
        (config.VERBOSE 의존성 제거)
        """

        if not prompts_boxes:
            # print/if config.VERBOSE -> logger.info
            self.logger.info("2단계: 분할할 프롬프트(BBox)가 없습니다. 빈 마스크를 반환합니다.")
            return np.zeros(image.shape[:2], dtype=bool)

        # print/if config.VERBOSE -> logger.info
        self.logger.info(f"2단계: {len(prompts_boxes)}개의 프롬프트로 정밀 분할 중...")

        # 1. BGR -> RGB 변환
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. SamPredictor에 이미지 설정
        self.predictor.set_image(rgb_image)

        # 3. 프롬프트(BBox)를 Numpy 배열로 변환
        input_boxes = np.array(prompts_boxes)

        # 4. 모델 추론
        masks, scores, logits = self.predictor.predict(
            box=input_boxes,
            multimask_output=False,
        )

        # 5. 결과 마스크 취합
        final_mask = np.any(masks, axis=0)

        # print/if config.VERBOSE -> logger.info
        self.logger.info(f"2단계: 정밀 마스크 생성 완료. (Shape: {final_mask.shape})")

        return final_mask


if __name__ == "__main__":

    # 테스트용 로거 가져오기
    test_logger = get_logger("SegmenterTest")
    test_logger.info("--- Segmenter 모듈 테스트 시작 ---")

    # 1. 설정 로드 및 디렉토리 생성
    try:
        config = Config()
        config.make_dirs()
    except Exception as e:
        test_logger.critical(f"Config 로드 실패: {e}")
        sys.exit(1)

    # 2. 테스트 이미지 생성 (임시)
    test_image = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 100), (300, 400), (255, 0, 0), -1)

    # 3. 테스트 BBox 프롬프트
    test_boxes = [
        [90, 90, 310, 410]
    ]

    test_logger.info(f"테스트 이미지 생성 (512x512) 및 BBox {test_boxes} 설정.")

    # 4. Segmenter 실행
    try:
        segmenter = Segmenter()
        mask = segmenter.segment(test_image, test_boxes)

        # 5. 결과 검증
        test_logger.info(f"반환된 마스크 정보: Shape={mask.shape}, Dtype={mask.dtype}")

        if mask.shape == (512, 512) and mask.dtype == bool:
            test_logger.info("✅ 테스트 성공. 마스크가 올바른 형태(Shape, Dtype)로 반환되었습니다.")

            # 6. 결과 시각화
            mask_visual = (mask * 255).astype(np.uint8)
            mask_colored = cv2.cvtColor(mask_visual, cv2.COLOR_GRAY2BGR)

            overlay = test_image.copy()
            overlay[mask] = (0, 0, 255)

            x1, y1, x2, y2 = test_boxes[0]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

            result_path = config.SAVE_RESULTS_DIR / "mask_test_segmenter.png"
            cv2.imwrite(str(result_path), overlay)
            test_logger.info(f"시각화 결과가 {result_path} 에 저장되었습니다.")

        else:
            test_logger.error("❌ 테스트 실패. 반환된 마스크가 올바른 형태가 아닙니다.")

    except Exception as e:
        test_logger.error("--- Segmenter 모듈 테스트 실패 ---")
        test_logger.error(f"오류: {e}")
        test_logger.error("SAM 모델 경로(SAM_MODEL_PATH)와 'segment-anything' 설치를 확인하세요.")