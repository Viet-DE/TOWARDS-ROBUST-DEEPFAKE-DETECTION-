import yaml
import os

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def merge_configs(base_config, override_config):
    """
    Gộp 2 dictionary config (đệ quy)
    override_config sẽ ghi đè lên base_config
    """
    for key, value in override_config.items():
        if isinstance(value, dict) and key in base_config:
            base_config[key] = merge_configs(base_config[key], value)
        else:
            base_config[key] = value
    return base_config

def get_config(config_path="configs/default.yaml", override_files=[]):
    """
    Load main config và merge thêm các config phụ (nếu có)
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
        
    config = load_yaml(config_path)
    
    # Load model-specific configs automatically if they exist in structure
    # (Optional logic: bạn có thể load thủ công ở ngoài)
    
    for path in override_files:
        if path and os.path.exists(path):
            print(f"Loading override config: {path}")
            overrides = load_yaml(path)
            config = merge_configs(config, overrides)
            
    return config