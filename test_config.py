"""
테스트용 설정 클래스
기존 실험 디렉토리를 사용하여 새로운 디렉토리를 생성하지 않습니다.
"""

from pathlib import Path
import torch


class TestConfig:
    """
    테스트용 설정 클래스
    - exp_dir: 기존 실험 디렉토리 경로 (필수)
    - data_dir: 테스트 데이터 디렉토리 경로 (필수)
    """
    
    def __init__(self, exp_dir: Path, data_dir: Path):
        """
        Args:
            exp_dir: 기존 실험 디렉토리 경로 (예: results/2025-11-07/exp1)
            data_dir: 테스트 데이터 디렉토리 경로
        """
        # 필수 경로 설정
        self.EXP_DIR = Path(exp_dir)
        self.data_dir = Path(data_dir)
        
        # exp_dir 검증
        if not self.EXP_DIR.exists():
            raise ValueError(f"실험 디렉토리가 존재하지 않습니다: {self.EXP_DIR}")
        
        # 결과 경로는 EXP_DIR 하위에 생성됩니다.
        self.LOG_DIR = self.EXP_DIR / "logs"
        self.VISUALIZATION_DIR = self.EXP_DIR / "visualizations"
        self.CHECKPOINT_DIR = self.EXP_DIR / "checkpoints"
        
        # 모델 경로 설정
        # Anomalib Engine은 default_root_dir 아래에 모델명 폴더를 생성하고 체크포인트를 저장합니다
        # 실제 저장 경로:
        # - checkpoints/Patchcore/latest/weight/lightning (또는 weights/lightning)
        # - checkpoints/Patchcore/v0/weights/lightning (또는 weight/lightning)
        self.ANOMALY_MODEL_NAME = "patchcore"
        
        # 체크포인트 찾기 우선순위:
        # 1. checkpoints/Patchcore/latest/weight/lightning (또는 weights/lightning)
        # 2. checkpoints/Patchcore/v0/weights/lightning (또는 weight/lightning)
        # 3. checkpoints/Patchcore/*/weight*/lightning (모든 버전)
        # 4. checkpoints/last.ckpt
        self._find_model_path()
        
        # PatchCore 모델 설정 (기본값)
        self.BACKBONE = "wide_resnet50_2"
        self.LAYERS = ('layer2', 'layer3')
        self.PRE_TRAINED = True
        self.CORESET_SAMPLING_RATIO = 0.1
        self.NUM_NEIGHBORS = 9
        self.PRE_PROCESSOR = True
        self.POST_PROCESSOR = True
        self.EVALUATOR = True
        self.VISUALIZER = True
        
        # 이미지 사이즈
        self.IMAGE_SIZE = (256, 256)
        
        # Device Settings
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.NUM_WORKERS = 0 if self.DEVICE == "cuda" else 4
        self.PIN_MEMORY = True if self.DEVICE == "cuda" else False
        self.BATCH_SIZE = 16 if self.DEVICE == "cuda" else 4
        
        # Miscellaneous
        self.VERBOSE = True
        self.DEBUG = False
        
        # 테스트 결과 디렉토리
        self.TEST_RESULTS_DIR = self.EXP_DIR / "test_results"
        self.TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    def _find_model_path(self):
        """
        체크포인트 디렉토리에서 모델 파일을 찾습니다.
        Anomalib은 다음 경로에 저장합니다:
        - checkpoints/Patchcore/latest/weight/lightning (또는 weights/lightning)
        - checkpoints/Patchcore/v0/weights/lightning (또는 weight/lightning)
        """
        patchcore_dir = self.CHECKPOINT_DIR / "Patchcore"
        
        if not patchcore_dir.exists():
            # Patchcore 디렉토리가 없으면 checkpoints/last.ckpt 확인
            last_ckpt = self.CHECKPOINT_DIR / "last.ckpt"
            if last_ckpt.exists():
                self.ANOMALY_MODEL_PATH = last_ckpt
                return
            self.ANOMALY_MODEL_PATH = None
            return
        
        # 우선순위 1: Patchcore/latest/weight/lightning 또는 weights/lightning
        latest_dir = patchcore_dir / "latest"
        if latest_dir.exists():
            for weight_dir_name in ["weight", "weights"]:
                weight_dir = latest_dir / weight_dir_name
                if weight_dir.exists():
                    lightning_file = weight_dir / "lightning"
                    if lightning_file.exists() and lightning_file.is_file():
                        self.ANOMALY_MODEL_PATH = lightning_file
                        return
                    # lightning이 디렉토리인 경우 내부 파일 찾기
                    if lightning_file.exists() and lightning_file.is_dir():
                        ckpt_files = list(lightning_file.glob("*.ckpt"))
                        if ckpt_files:
                            latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
                            self.ANOMALY_MODEL_PATH = latest_ckpt
                            return
        
        # 우선순위 2: Patchcore/v0/weights/lightning 또는 weight/lightning
        v0_dir = patchcore_dir / "v0"
        if v0_dir.exists():
            for weight_dir_name in ["weight", "weights"]:
                weight_dir = v0_dir / weight_dir_name
                if weight_dir.exists():
                    lightning_file = weight_dir / "lightning"
                    if lightning_file.exists() and lightning_file.is_file():
                        self.ANOMALY_MODEL_PATH = lightning_file
                        return
                    # lightning이 디렉토리인 경우 내부 파일 찾기
                    if lightning_file.exists() and lightning_file.is_dir():
                        ckpt_files = list(lightning_file.glob("*.ckpt"))
                        if ckpt_files:
                            latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
                            self.ANOMALY_MODEL_PATH = latest_ckpt
                            return
        
        # 우선순위 3: Patchcore/*/weight*/lightning (모든 버전)
        for version_dir in patchcore_dir.iterdir():
            if version_dir.is_dir():
                for weight_dir_name in ["weight", "weights"]:
                    weight_dir = version_dir / weight_dir_name
                    if weight_dir.exists():
                        lightning_file = weight_dir / "lightning"
                        if lightning_file.exists() and lightning_file.is_file():
                            self.ANOMALY_MODEL_PATH = lightning_file
                            return
                        # lightning이 디렉토리인 경우 내부 파일 찾기
                        if lightning_file.exists() and lightning_file.is_dir():
                            ckpt_files = list(lightning_file.glob("*.ckpt"))
                            if ckpt_files:
                                latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
                                self.ANOMALY_MODEL_PATH = latest_ckpt
                                return
        
        # 우선순위 4: checkpoints/last.ckpt
        last_ckpt = self.CHECKPOINT_DIR / "last.ckpt"
        if last_ckpt.exists():
            self.ANOMALY_MODEL_PATH = last_ckpt
            return
        
        # 우선순위 5: Patchcore 디렉토리 내의 모든 .ckpt 파일
        all_ckpt_files = list(patchcore_dir.rglob("*.ckpt"))
        if all_ckpt_files:
            latest_ckpt = max(all_ckpt_files, key=lambda p: p.stat().st_mtime)
            self.ANOMALY_MODEL_PATH = latest_ckpt
            return
        
        # 모델을 찾지 못한 경우 None으로 설정
        self.ANOMALY_MODEL_PATH = None

