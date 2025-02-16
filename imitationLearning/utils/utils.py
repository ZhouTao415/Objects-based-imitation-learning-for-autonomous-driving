import os
import yaml

def make_abs_path(root_file: str, *rel_paths: str) -> str:
    """ Convert relative paths to absolute paths, 
    supporting concatenation of multiple paths """
    root_dir = os.path.dirname(os.path.abspath(root_file))
    abs_path = os.path.abspath(os.path.join(root_dir, *rel_paths))
    return os.path.normpath(abs_path)  # Normalize the path to avoid redundant ".." references

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config