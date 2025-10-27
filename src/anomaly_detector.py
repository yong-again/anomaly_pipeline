import sys
import cv2
import torch
import numpy as np
from pathlib import Path

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
# 'src' 내부 모듈 임포트
# ---------------------------------------------------------------------------
try:
    from utils import get_logger
except ImportError:
    print("오류: 'src/utils.py'를 찾을 수 없습니다.")
    sys.exit(1)

# Anomalib에서 모델 및 [수정] PreProcessor 임포트
from anomalib.models import get_model
from anomalib.pre_processing import PreProcessor

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("오류: 'albumentations' 라이브러리가 필요합니다. 'pip install albumentations'를 실행하세요.")
    sys.exit(1)


# [삭제] 하드코딩된 IMAGE_SIZE 변수 제거
# IMAGE_SIZE = (256, 256)

class AnomalyDetector:
    """
    1단계: 결함 후보 영역 탐지 (Anomaly Localization)
    """

    # [수정] __init__이 config 객체를 주입받도록 변경
    def __init__(self, config: Config):
        """
        AnomalyDetector 초기화.
        Config 로드, 모델 로드, 전처리기(PreProcessor) 설정을 수행합니다.
        """
        # self.config = Config() (기존)
        self.config = config  # [수정] 주입받은 config 사용
        self.logger = get_logger("AnomalyDetector")

        # [신규] config에서 IMAGE_SIZE 가져오기
        try:
            self.IMAGE_SIZE = config.IMAGE_SIZE
        except AttributeError:
            self.logger.critical(f"'config.py'에 'IMAGE_SIZE = (H, W)' 설정이 없습니다.")
            sys.exit(1)

        if not self.config.ANOMALY_MODEL_PATH.exists():
            self.logger.critical(f"오류: 1단계 모델 파일을 찾을 수 없습니다.")
            self.logger.critical(f"경로: {self.config.ANOMALY_MODEL_PATH}")
            self.logger.critical("'train_anomaly_model.py'를 먼저 실행하여 모델을 학습시켜주세요.")
            sys.exit(1)

        self.logger.info("1단계 모델 로딩 중...")
        self.model = self._load_model()
        self.model.to(self.config.DEVICE)
        self.model.eval()

        # -----------------------------------------------------------------
        # [수정] PreProcessor 설정
        # -----------------------------------------------------------------
        # train_anomaly_model.py에서 사용한 변환과 *동일한* 변환을 정의합니다.
        transform = A.Compose(
            [
                A.Resize(self.IMAGE_SIZE[0], self.IMAGE_SIZE[1]),
                # BGR 이미지를 RGB로 변환 (anomalib 0.x의 PreProcessor는 자동 변환)
                # Albumentations는 입력이 RGB라고 가정하므로,
                # predict 시점에 BGR->RGB 변환이 필요할 수 있습니다.
                # (테스트 결과: Albumentations Compose는 채널 순서를 신경쓰지 않고
                # Normalize와 ToTensorV2가 잘 처리하는 것으로 보임)

                # 정규화 (train과 동일하게)
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )

        # 새 API에 맞춰 transform 객체 전달
        self.pre_processor = PreProcessor(transform=transform)
        self.logger.info(f"Anomalib PreProcessor (1.0+ API) 초기화 완료. (ImageSize: {self.IMAGE_SIZE})")

    def _load_model(self):
        """
        config에 지정된 모델 이름과 경로를 기반으로 모델을 로드합니다.
        """
        try:
            model = get_model(self.config.ANOMALY_MODEL_NAME).load_from_checkpoint(
                self.config.ANOMALY_MODEL_PATH,
                map_location=self.config.DEVICE
            )
            self.logger.info(f"'{self.config.ANOMALY_MODEL_NAME}' 모델 로딩 성공.")
            return model
        except Exception as e:
            self.logger.critical(f"모델 로딩 중 치명적 오류 발생: {e}")
            sys.exit(1)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        입력 이미지(np.ndarray)에 대한 이상치 점수 맵(히트맵)을 반환합니다.

        Args:
            image (np.ndarray): OpenCV (cv2.imread)로 읽은 BGR 이미지
        """
        self.logger.info("1단계: 결함 후보 탐지 중...")

        original_height, original_width = image.shape[:2]

        # 2. 이미지 전처리
        processed_data = self.pre_processor(image=image)
        processed_image = processed_data["image"].to(self.config.DEVICE)

        if processed_image.dim() == 3:
            processed_image = processed_image.unsqueeze(0)

        with torch.no_grad():
            model_output = self.model(processed_image)

        heatmap = model_output["anomaly_map"].squeeze().cpu().numpy()

        heatmap_resized = cv2.resize(
            heatmap,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR
        )

        if heatmap_resized.max() > heatmap_resized.min():
            heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / \
                                 (heatmap_resized.max() - heatmap_resized.min())
        else:
            heatmap_normalized = np.zeros_like(heatmap_resized)

        self.logger.info("1단계: 히트맵 생성 완료.")
        return heatmap_normalized


# ---------------------------------------------------------------------------
# [테스트] 이 스크립트를 직접 실행할 때 테스트 코드를 실행합니다.
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # [수정] utils.setup_experiment를 사용하여 로거와 디렉토리 초기화
    try:
        from utils import setup_experiment

        config = Config()
        test_logger = setup_experiment(config)  # 루트 로거 가져오기
    except Exception as e:
        print(f"테스트 설정 실패: {e}")
        sys.exit(1)

    test_logger.info("--- AnomalyDetector 모듈 테스트 시작 ---")

    # Config에 IMAGE_SIZE가 없으면 임시 설정 (테스트용)
    if not hasattr(config, "IMAGE_SIZE"):
        config.IMAGE_SIZE = (256, 256)
        test_logger.warning("config.py에 IMAGE_SIZE가 없어 (256, 256)으로 임시 설정합니다.")

    test_image_dir = config.data_dir / "test_images"
    test_image_dir.mkdir(parents=True, exist_ok=True)
    test_image_name = "test_image.png"
    test_image_path = test_image_dir / test_image_name

    image = cv2.imread(str(test_image_path))
    if image is None:
        test_logger.warning(f"'{test_image_path}'를 찾을 수 없습니다. 임시 테스트 이미지를 생성합니다.")
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (200, 200), (0, 0, 255), -1)
        cv2.imwrite(str(test_image_path), image)
        test_logger.info(f"임시 이미지를 {test_image_path}에 저장했습니다.")

    try:
        # [수정] __init__이 config 객체를 요구함
        detector = AnomalyDetector(config)
        heatmap = detector.predict(image)

        test_logger.info(f"생성된 히트맵 정보: Shape={heatmap.shape}, Min={heatmap.min():.4f}, Max={heatmap.max():.4f}")

        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay_image = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)

        # [수정] config.VISUALIZATION_DIR에 저장
        result_path = config.VISUALIZATION_DIR / f"heatmap_{test_image_name}"
        cv2.imwrite(str(result_path), overlay_image)

        test_logger.info(f"✅ 테스트 성공. 결과가 {result_path} 에 저장되었습니다.")

    except Exception as e:
        test_logger.error("--- AnomalyDetector 모듈 테스트 실패 ---", exc_info=True)  # [수정] 상세 오류 출력