import yaml

def load_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)

def get_paths():
    return load_config("config/paths.yaml")
