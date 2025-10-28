"""
Custom exception hierarchy for Drug-Gene CDSS.
Provides clear, actionable error messages.
"""

class CDSSException(Exception):
    """Base exception for all CDSS errors"""
    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)
    
    def __str__(self):
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class DataFetchException(CDSSException):
    """Raised when external API data fetching fails"""
    pass


class FeaturizationException(CDSSException):
    """Raised when molecular/protein featurization fails"""
    pass


class ModelException(CDSSException):
    """Raised when model loading or inference fails"""
    pass


class ValidationException(CDSSException):
    """Raised when input validation fails"""
    pass


class ConfigurationException(CDSSException):
    """Raised when configuration is invalid"""
    pass


# User-friendly error messages
ERROR_MESSAGES = {
    'smiles_not_found': 'Could not find molecular structure for drug "{drug_id}". Please verify the identifier.',
    'sequence_not_found': 'Could not find protein sequence for gene "{gene_id}". Please verify the identifier.',
    'invalid_smiles': 'The molecular structure (SMILES: {smiles}) is invalid or corrupt.',
    'conformer_failed': 'Failed to generate 3D structure for molecule. The chemical structure may be too complex.',
    'model_not_found': 'Model weights file not found at {path}. Please download the trained model.',
    'cuda_unavailable': 'GPU not available. Running on CPU (slower performance expected).',
    'api_rate_limit': 'API rate limit exceeded for {service}. Please wait {retry_after} seconds.',
    'cache_corrupted': 'Cache file corrupted: {path}. It will be regenerated.',
}


def format_user_error(error_key: str, **kwargs) -> str:
    """Format user-friendly error message"""
    template = ERROR_MESSAGES.get(error_key, "An unexpected error occurred.")
    return template.format(**kwargs)
