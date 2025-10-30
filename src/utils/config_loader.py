import yaml
import os
import re
from pathlib import Path
from typing import Dict, Any

def replace_env_var(match):
    var_name = match.group(1)
    default_value = match.group(2)
    
    value = os.getenv(var_name, default_value)
    
    if value is None:
        raise ValueError(
            f"Environment variable '{var_name}' not set and no default provided"
        )
    return value

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration with environment variable substitution.
    The syntax for substitution is: 
        ${VAR_NAME} or ${VAR_NAME:default_value}
    
    Parameters:
        config_path (str): Path to the YAML config file. Defaults to "config/config.yaml".
        
    Raises:
        ValueError: If an environment variable is not set and no default is provided.

    Returns:
        Dict[str, Any]: The configuration parameters loaded from the YAML file.
    """
    # Ensure the config path exists
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"The config file '{config_path}' does not exist.")

    with open(config_file, 'r') as f:
        config_text = f.read()

    # Regex pattern to find placeholders in the config
    pattern = re.compile(r'\$\{([^}:]+)(?::([^}]+))?\}')
    config_text = re.sub(pattern, replace_env_var, config_text)
    
    # Parse the modified config text as YAML
    config = yaml.safe_load(config_text)
    
    return config
