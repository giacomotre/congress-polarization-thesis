from src.config_loader import load_config, get_paths

if __name__ == "__main__":
    svm_cfg = load_config("config/svm_config.yaml")
    paths = get_paths()

    print("✅ SVM Config Loaded:")
    print(svm_cfg)

    print("\n📁 Paths Loaded:")
    print(paths)
