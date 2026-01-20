"""
NextGen Enterprise API Integration Module
"""

from .nextgen_client import (
    NextGenAPIConfig,
    NextGenAPIClient,
    CRCAuditWorkflow,
    CRC_LOINC_CODES,
    CRC_CPT_CODES,
    CRC_DOCUMENT_KEYWORDS
)

__all__ = [
    'NextGenAPIConfig',
    'NextGenAPIClient', 
    'CRCAuditWorkflow',
    'CRC_LOINC_CODES',
    'CRC_CPT_CODES',
    'CRC_DOCUMENT_KEYWORDS'
]
