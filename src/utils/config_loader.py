import yaml
import os
from pathlib import Path
from typing import Dict, Any
import re


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load config with environment variable substitution.
    
    Syntax: ${VAR_NAME} or ${VAR_NAME:default_value}
    """
    with open(config_path, 'r') as f:
        config_text = f.read()
    
    # Find all ${VAR} or ${VAR:default}
    pattern = r'\$\{([^}:]+)(?::([^}]+))?\}'
    
    def replace_env_var(match):
        var_name = match.group(1)
        default_value = match.group(2)
        
        value = os.getenv(var_name, default_value)
        
        if value is None:
            raise ValueError(
                f"Environment variable '{var_name}' not set and no default provided"
            )
        
        return value
    
    config_text = re.sub(pattern, replace_env_var, config_text)
    
    config = yaml.safe_load(config_text)
    
    return config
