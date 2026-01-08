"""UDS LayoutLM - HRSA UDS Metrics Extraction using LayoutLMv3."""

from .labels import UDS_LABELS, LABEL2ID, ID2LABEL, UDS_MEASURES
from .processor import PDFProcessor, ProcessedPage
from .dataset import create_dataset_from_labeled, load_dataset
from .inference import UDSExtractor, ExtractedEntity, ExtractionResult

__version__ = "0.1.0"
__all__ = [
    "UDS_LABELS",
    "LABEL2ID", 
    "ID2LABEL",
    "UDS_MEASURES",
    "PDFProcessor",
    "ProcessedPage",
    "create_dataset_from_labeled",
    "load_dataset",
    "UDSExtractor",
    "ExtractedEntity",
    "ExtractionResult",
]