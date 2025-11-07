import multiprocessing
import torch
from pathlib import Path
import datetime
import re


class Config:
    def __init__(self):
        cpu_count = multiprocessing.cpu_count()

        # -----------
        # Data Paths
        # -----------
        category = 'cable'
        self.data_dir = Path.cwd() / "data" 
        self.NORMAL_DATA_DIR = self.data_dir / "normal_dir"
        self.ANOMALY_DATA_DIR = self.data_dir / "abnormal_dir"
        self.MASK_DATA_DIR = self.data_dir / "mask_dir"


        # -------------
        # [수정] 기본 모델 및 결과 경로
        # -------------
        # 'results' 폴더가 모든 실험의 상위 폴더가 됩니다.
        self.BASE_RESULTS_DIR = Path.cwd() / "results"

        # 'models' 폴더는 SAM 같은 사전 학습된 모델만 보관합니다.
        self.PRETRAINED_MODEL_DIR = Path.cwd() / "models"
        self.SAM_MODEL_PATH = self.PRETRAINED_MODEL_DIR / "sam_vit_h_4b8939.pth"

        # [신규] 실험 경로 생성
        # e.g., results/2025-10-24/exp1
        self.EXP_DIR = self._create_experiment_dir(self.BASE_RESULTS_DIR)

        # 결과 경로는 EXP_DIR 하위에 생성됩니다.
        self.LOG_DIR = self.EXP_DIR / "logs"
        self.VISUALIZATION_DIR = self.EXP_DIR / "visualizations"
        self.CHECKPOINT_DIR = self.EXP_DIR / "checkpoints"  # 체크포인트 저장 위치

        # [수정] 모델 경로가 CHECKPOINT_DIR를 바라보도록 변경
        self.ANOMALY_MODEL_PATH = self.CHECKPOINT_DIR / "anomaly_model"
        self.ANOMALY_MODEL_NAME = "cfa"

        # [추가] 이미지 사이즈
        self.IMAGE_SIZE = (256, 256)

        # ----------------
        # Thresholds
        # ----------------
        self.HEATMAP_THRESHOLD = 0.75

        # ----------------
        # Device Settings
        # ----------------
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.NUM_WORKERS = cpu_count if self.DEVICE == "cpu" else 0
        self.PIN_MEMORY = True if self.DEVICE == "cuda" else False
        self.BATCH_SIZE = 16 if self.DEVICE == "cuda" else 4
        self.NUM_EPOCHS = 50 if self.DEVICE == "cuda" else 10
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 1e-5
        self.SAVE_MODEL_EVERY = 5
        self.LOAD_MODEL = False
        # [수정] 이 경로도 CHECKPOINT_DIR 하위로 변경
        self.SAVE_MODEL_PATH = self.CHECKPOINT_DIR / "best_model.pth"

        # ----------------
        # Miscellaneous
        # ----------------
        self.RANDOM_SEED = 42
        # self.LOG_DIR (위에서 정의됨)
        # self.SAVE_RESULTS_DIR (VISUALIZATION_DIR로 대체됨)
        self.VERBOSE = True
        self.DEBUG = False
        self.AUGMENT_DATA = True

        # [중요] Config 생성 시점에 make_dirs()를 호출하지 않고,
        # setup_experiment_paths (utils.py)에서 명시적으로 호출하도록 변경합니다.
        # self.make_dirs()

        if self.VERBOSE:
            print(f"Configuration initialized. Using device: {self.DEVICE}")
            print(f"✅ 실험 경로 생성: {self.EXP_DIR}")

    def _create_experiment_dir(self, base_results_dir: Path) -> Path:
        """ 'base_results_dir / YYYY-MM-DD / exp{N}' 형태의 디렉토리를 생성합니다. """
        try:
            # 1. 날짜 폴더 (e.g., .../results/2025-10-24)
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            date_dir = base_results_dir / today
            date_dir.mkdir(parents=True, exist_ok=True)

            # 2. 'exp{N}' 번호 찾기
            existing_exps = list(date_dir.glob('exp*'))
            next_exp_num = 1
            if existing_exps:
                max_num = 0
                for exp_path in existing_exps:
                    match = re.search(r'exp(\d+)', exp_path.name)
                    if match:
                        max_num = max(max_num, int(match.group(1)))
                next_exp_num = max_num + 1

            # 3. 새 실험 폴더 경로 (e.g., .../results/2025-10-24/exp1)
            new_exp_dir = date_dir / f"exp{next_exp_num}"

            return new_exp_dir

        except Exception as e:
            print(f"실험 디렉토리 생성 실패: {e}")
            # 오류 발생 시 기본 'results' 폴더 사용
            return base_results_dir

    def make_dirs(self):
        """ [수정] EXP_DIR 하위의 모든 필수 디렉토리를 생성합니다. """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.PRETRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # [신규] 새 실험 경로 하위 폴더 생성
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        if self.VERBOSE:
            print(f"모든 출력 디렉토리 생성 완료: {self.EXP_DIR}")