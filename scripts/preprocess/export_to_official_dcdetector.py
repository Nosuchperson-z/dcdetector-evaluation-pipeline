import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data_raw"
PROCESSED_ROOT = ROOT / "data_processed"
OFFICIAL_DATASET_ROOT = ROOT / "KDD2023-DCdetector" / "dataset"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    print(f"{src} -> {dst}")


def export_smap():
    src_root = RAW_ROOT / "SMAP"
    dst_root = OFFICIAL_DATASET_ROOT / "SMAP"
    copy_file(src_root / "SMAP_train.npy", dst_root / "SMAP_train.npy")
    copy_file(src_root / "SMAP_test.npy", dst_root / "SMAP_test.npy")
    copy_file(src_root / "SMAP_test_label.npy", dst_root / "SMAP_test_label.npy")


def export_msl():
    src_root = RAW_ROOT / "MSL"
    dst_root = OFFICIAL_DATASET_ROOT / "MSL"
    copy_file(src_root / "MSL_train.npy", dst_root / "MSL_train.npy")
    copy_file(src_root / "MSL_test.npy", dst_root / "MSL_test.npy")
    copy_file(src_root / "MSL_test_label.npy", dst_root / "MSL_test_label.npy")


def export_processed_dataset(dataset_name: str):
    src_root = PROCESSED_ROOT / dataset_name
    dst_root = OFFICIAL_DATASET_ROOT / dataset_name
    copy_file(src_root / "train.npy", dst_root / "train.npy")
    copy_file(src_root / "val.npy", dst_root / "val.npy")
    copy_file(src_root / "test.npy", dst_root / "test.npy")
    copy_file(src_root / "label.npy", dst_root / "label.npy")
    copy_file(src_root / "meta.json", dst_root / "meta.json")


def main():
    export_smap()
    export_msl()
    export_processed_dataset("HAI21.03")
    print("Exported SMAP/MSL raw arrays and HAI21.03 processed arrays to official dataset/.")


if __name__ == "__main__":
    main()
