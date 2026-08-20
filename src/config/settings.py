import yaml
from pathlib import Path

def load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as s:
        return yaml.safe_load(s)
    
    


