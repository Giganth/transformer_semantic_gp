from copy import deepcopy
from datetime import datetime
import os
import yaml
from .. import __version__

class ConfigLoader:
    """Loads configuration from multiple sources."""
    
    @classmethod
    def load(cls, config_path: str = None, algorithm: str = None) -> dict:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to YAML config file (optional)
            algorithm: Algorithm type ('standard_gp' or 'transformer_gp')
        
        Returns:
            Configuration dictionary
        """
        # Start with algorithm-specific defaults
        if algorithm == 'transformer_gp':
            config = cls._get_transformer_defaults()
        elif algorithm == 'standard_gp':
            config = cls._get_standard_defaults()
        elif algorithm == 'data_generator':
            config = cls._get_data_generator_defaults()
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm}")
        
        # Load symbolic regression defaults as base, then layer algorithm defaults on top
        sr_defaults = cls._get_symbolic_regression_defaults()
        config = cls._deep_update(sr_defaults, config)

        # Merge with user configuration if provided
        if config_path and os.path.exists(config_path):
            user_config = cls._load_yaml(config_path)
            config = cls._deep_update(config, user_config)
        
        # Add metadata
        config['metadata'] = {
            'version': __version__,
            'created_at': str(datetime.now()),
            'config_path': config_path or 'defaults'
        }
        
        return config

    @staticmethod
    def _get_standard_defaults() -> dict:
        from TSGP.config.defaults import STANDARD_GP_DEFAULTS
        return deepcopy(STANDARD_GP_DEFAULTS)
    
    @staticmethod
    def _get_transformer_defaults() -> dict:
        from TSGP.config.defaults import TS_GP_DEFAULTS
        return deepcopy(TS_GP_DEFAULTS)
    
    @staticmethod
    def _get_data_generator_defaults() -> dict:
        from TSGP.config.defaults import DATA_GENERATOR_DEFAULTS
        return deepcopy(DATA_GENERATOR_DEFAULTS)
    
    @staticmethod
    def _get_symbolic_regression_defaults() -> dict:
        from TSGP.config.defaults import SYMBOLIC_REGRESSION_DEFAULTS
        return deepcopy(SYMBOLIC_REGRESSION_DEFAULTS)
    
    @staticmethod
    def _load_yaml(path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def _deep_update(target: dict, update: dict) -> dict:
        """Recursively update a dictionary."""
        for key, value in update.items():
            if isinstance(value, dict) and key in target:
                target[key] = ConfigLoader._deep_update(target[key], value)
            else:
                target[key] = value
        return target