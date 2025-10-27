import argparse
import shutil
from pathlib import Path
import logging

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def reorganize_mvtec_category(input_dir: Path, output_dir: Path):
    """
    MVTec AD의 단일 카테고리 폴더 구조를
    (normal_dir, abnormal_dir, mask_dir) 3개 폴더로 재구성합니다.
    """

    if not input_dir.is_dir():
        logger.error(f"입력 경로를 찾을 수 없습니다: {input_dir}")
        return

    # 1. 소스(Source) 경로 정의
    normal_src_dir = input_dir / "train" / "good"
    abnormal_src_root = input_dir / "test"
    mask_src_root = input_dir / "ground_truth"

    # 2. 대상(Destination) 경로 정의
    normal_dest_dir = output_dir / "normal_dir"
    abnormal_dest_dir = output_dir / "abnormal_dir"
    mask_dest_dir = output_dir / "mask_dir"

    # 3. 대상 폴더 생성
    logger.info(f"출력 폴더를 생성합니다: {output_dir}")
    normal_dest_dir.mkdir(parents=True, exist_ok=True)
    abnormal_dest_dir.mkdir(parents=True, exist_ok=True)
    mask_dest_dir.mkdir(parents=True, exist_ok=True)

    # 4. 파일 복사: Normal (Train)
    if not normal_src_dir.exists():
        logger.warning(f"정상 이미지 폴더를 찾을 수 없습니다: {normal_src_dir}")
    else:
        logger.info(f"'{normal_src_dir.name}'의 정상 이미지 복사 중...")
        count = 0
        for file_path in normal_src_dir.glob("*.png"):
            shutil.copy(file_path, normal_dest_dir / file_path.name)
            count += 1
        logger.info(f"✅ 총 {count}개의 정상 이미지를 {normal_dest_dir.name}로 복사했습니다.")

    # 5. 파일 복사: Abnormal (Test) 및 Masks
    if not abnormal_src_root.exists():
        logger.warning(f"테스트 이미지 폴더를 찾을 수 없습니다: {abnormal_src_root}")
        return

    abnormal_count = 0
    mask_count = 0

    # 'test' 폴더 하위의 모든 결함 유형 디렉토리를 순회
    for defect_type_dir in abnormal_src_root.iterdir():
        if defect_type_dir.is_dir() and defect_type_dir.name != "good":
            defect_type = defect_type_dir.name
            logger.info(f"결함 유형 '{defect_type}' 처리 중...")

            # (A) 비정상 이미지 복사
            for image_path in defect_type_dir.glob("*.png"):
                shutil.copy(image_path, abnormal_dest_dir / image_path.name)
                abnormal_count += 1

            # (B) 마스크 이미지 복사
            mask_src_dir = mask_src_root / defect_type
            if mask_src_dir.exists():
                for mask_path in mask_src_dir.glob("*.png"):
                    shutil.copy(mask_path, mask_dest_dir / mask_path.name)
                    mask_count += 1
            else:
                logger.warning(f"  '{defect_type}'에 해당하는 마스크 폴더가 없습니다: {mask_src_dir}")

    logger.info(f"✅ 총 {abnormal_count}개의 비정상 이미지를 {abnormal_dest_dir.name}로 복사했습니다.")
    logger.info(f"✅ 총 {mask_count}개의 마스크를 {mask_dest_dir.name}로 복사했습니다.")
    logger.info("데이터 재구성 완료.")


# --- 스크립트 실행 부분 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MVTec AD 데이터셋 카테고리를 'normal/abnormal/mask' 구조로 재구성합니다.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=True,
        help="MVTec AD의 원본 카테고리 경로.\n"
             "예: /path/to/mvtec_ad/bottle"
    )

    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="재구성된 데이터가 저장될 새로운 루트 경로.\n"
             "예: /path/to/my_custom_dataset/bottle"
    )

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    reorganize_mvtec_category(input_path, output_path)