"""Inference module for UDS metrics extraction."""

import os
import importlib
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict
import json

import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from PIL import Image
from tqdm import tqdm

# Suppress deprecated device argument warning from transformers internal code
warnings.filterwarnings("ignore", message=".*device.*argument is deprecated.*", category=FutureWarning)

from .processor import PDFProcessor, ProcessedPage


# CRC Triage measure groupings
CRC_TRIAGE_MEASURES = {
    "colonoscopy": ["DOC_TYPE_COLONOSCOPY", "PROCEDURE_DATE", "INDICATION_SCREENING", "INDICATION_SURVEILLANCE", "INDICATION_DIAGNOSTIC"],
    "fit_test": ["DOC_TYPE_FIT", "COLLECTION_DATE", "RESULT_NEGATIVE", "RESULT_POSITIVE", "RESULT_VALUE"],
    "fobt_test": ["DOC_TYPE_FOBT", "COLLECTION_DATE", "RESULT_NEGATIVE", "RESULT_POSITIVE", "RESULT_VALUE"],
    "sigmoidoscopy": ["DOC_TYPE_SIGMOIDOSCOPY", "PROCEDURE_DATE"],
    "ct_colonography": ["DOC_TYPE_CT_COLONOGRAPHY", "PROCEDURE_DATE"],
    "polyp_findings": ["POLYP_FINDING", "POLYP_LOCATION", "POLYP_SIZE", "POLYP_COUNT", "PATHOLOGY_DIAGNOSIS", "BIOPSY_TAKEN", "BIOPSY_RESULT"],
    "other_findings": ["DIVERTICULA_FINDING", "HEMORRHOIDS_FINDING", "COMPLICATIONS"],
}

# Mammogram triage entity types - used to determine if document is a mammogram report
# Requires >= 2 different entity types to avoid FPs from documents that just mention "mammogram"
MAMMOGRAM_TRIAGE_TYPES = {'EXAM_MAMMOGRAM', 'EXAM_TOMOSYNTHESIS', 'BIRADS_CATEGORY', 'BREAST_DENSITY'}


def is_mammogram_report(entities: List['ExtractedEntity'], min_entity_types: int = 2) -> bool:
    """
    Determine if a document is a mammogram report based on extracted entities.
    
    Uses multi-entity rule: A true mammogram report should have at least 2 different
    mammogram-related entity types (e.g., EXAM_MAMMOGRAM + BIRADS_CATEGORY).
    This prevents false positives from documents that just mention "mammogram" in text.
    
    Args:
        entities: List of extracted entities from the document
        min_entity_types: Minimum number of different entity types required (default: 2)
        
    Returns:
        True if document appears to be a mammogram report
    """
    mmg_entity_types = {e.entity_type for e in entities if e.entity_type in MAMMOGRAM_TRIAGE_TYPES}
    return len(mmg_entity_types) >= min_entity_types


@dataclass
class ExtractedEntity:
    """An extracted entity from a document."""
    entity_type: str
    text: str
    confidence: float
    page: int
    bbox: List[int]  # Normalized 0-1000
    bbox_pixels: Optional[List[int]] = None  # Original pixel coordinates


@dataclass 
class ExtractionResult:
    """Complete extraction result for a document."""
    source_file: str
    num_pages: int
    entities: List[ExtractedEntity]
    uds_metrics: Dict[str, List[ExtractedEntity]]
    
    def to_dict(self) -> Dict:
        return {
            "source_file": self.source_file,
            "num_pages": self.num_pages,
            "entities": [asdict(e) for e in self.entities],
            "uds_metrics": {
                k: [asdict(e) for e in v] 
                for k, v in self.uds_metrics.items()
            }
        }
    
    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class UDSExtractor:
    """Extract UDS metrics from clinical documents."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        confidence_threshold: float = 0.7,
        labels_module: str = "src.labels_crc_triage"
    ):
        """
        Initialize the extractor.
        
        Args:
            model_path: Path to fine-tuned model
            device: Device to run inference on ("cuda" or "cpu")
            confidence_threshold: Minimum confidence for entity extraction
            labels_module: Module containing ID2LABEL mapping (e.g., "src.labels_crc_triage")
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        
        # Load labels dynamically
        labels = importlib.import_module(labels_module)
        self.id2label = labels.ID2LABEL
        self.uds_measures = getattr(labels, 'UDS_MEASURES', CRC_TRIAGE_MEASURES)
        
        print(f"Loading model from {model_path}...")
        self.processor = LayoutLMv3Processor.from_pretrained(
            model_path,
            apply_ocr=False
        )
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            model_path
        ).to(self.device)
        self.model.eval()
        
        self.pdf_processor = PDFProcessor()
        print(f"Model loaded. Using device: {self.device}")
        print(f"Labels: {len(self.id2label)} classes from {labels_module}")
    
    def extract_from_pdf(self, pdf_path: str) -> ExtractionResult:
        """Extract all UDS entities from a PDF document."""
        pages = self.pdf_processor.process_pdf(pdf_path)
        
        all_entities = []
        for page in pages:
            entities = self._extract_from_page(page)
            all_entities.extend(entities)
        
        # Group by UDS measures
        uds_metrics = self._group_by_uds_measure(all_entities)
        
        return ExtractionResult(
            source_file=pdf_path,
            num_pages=len(pages),
            entities=all_entities,
            uds_metrics=uds_metrics
        )
    
    def extract_from_image(
        self, 
        image: Union[str, Image.Image],
        words: Optional[List[str]] = None,
        boxes: Optional[List[List[int]]] = None
    ) -> List[ExtractedEntity]:
        """Extract entities from a single image."""
        if isinstance(image, str):
            page = self.pdf_processor.process_image(image)
        else:
            if words is None or boxes is None:
                raise ValueError("Must provide words and boxes for PIL Image")
            page = ProcessedPage(
                image=image,
                words=words,
                boxes=boxes,
                page_num=0,
                source_file="",
                raw_boxes=boxes
            )
        
        return self._extract_from_page(page)
    
    def _extract_from_page(self, page: ProcessedPage) -> List[ExtractedEntity]:
        """Extract entities from a single processed page."""
        if not page.words:
            return []
        
        # Prepare input
        encoding = self.processor(
            page.image,
            page.words,
            boxes=page.boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
            return_offsets_mapping=False
        )
        
        # Get word_ids BEFORE converting to dict (which loses the method)
        word_ids = encoding.word_ids(0)
        
        # Now move tensors to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        # Get predictions and confidences
        probs = torch.softmax(outputs.logits, dim=-1)
        predictions = probs.argmax(-1).squeeze().tolist()
        confidences = probs.max(-1).values.squeeze().tolist()
        
        # Handle single prediction case
        if isinstance(predictions, int):
            predictions = [predictions]
            confidences = [confidences]
        
        # Extract entities using BIO tags
        # Track last word_idx to handle subword tokens (multiple tokens per word)
        entities = []
        current_entity_type = None
        current_words = []
        current_word_indices = set()  # Track which word indices we've added
        current_boxes = []
        current_raw_boxes = []
        current_confs = []
        last_word_idx = None
        
        for idx, (pred, conf) in enumerate(zip(predictions, confidences)):
            # Get word index
            if word_ids is not None:
                word_idx = word_ids[idx]
                if word_idx is None:
                    continue
            else:
                word_idx = idx
                if word_idx >= len(page.words):
                    continue
            
            label = self.id2label.get(pred, "O")
            
            if label.startswith("B-"):
                entity_type = label[2:]
                
                # If same word and same entity type, skip (subword token)
                if word_idx == last_word_idx and current_entity_type == entity_type:
                    current_confs.append(conf)  # Accumulate confidence
                    continue
                
                # Save previous entity if exists
                if current_entity_type and current_words:
                    entity = self._create_entity(
                        current_entity_type,
                        current_words,
                        current_boxes,
                        current_raw_boxes,
                        current_confs,
                        page.page_num
                    )
                    if entity.confidence >= self.confidence_threshold:
                        entities.append(entity)
                
                # Start new entity
                current_entity_type = entity_type
                current_words = [page.words[word_idx]]
                current_word_indices = {word_idx}
                current_boxes = [page.boxes[word_idx]]
                current_raw_boxes = [page.raw_boxes[word_idx]] if page.raw_boxes else []
                current_confs = [conf]
                last_word_idx = word_idx
                
            elif label.startswith("I-"):
                entity_type = label[2:]
                if current_entity_type == entity_type:
                    # Only add word if we haven't seen this word_idx yet
                    if word_idx not in current_word_indices:
                        current_words.append(page.words[word_idx])
                        current_word_indices.add(word_idx)
                        current_boxes.append(page.boxes[word_idx])
                        if page.raw_boxes:
                            current_raw_boxes.append(page.raw_boxes[word_idx])
                    current_confs.append(conf)
                    last_word_idx = word_idx
            else:
                # "O" label - save current entity if exists
                if current_entity_type and current_words:
                    entity = self._create_entity(
                        current_entity_type,
                        current_words,
                        current_boxes,
                        current_raw_boxes,
                        current_confs,
                        page.page_num
                    )
                    if entity.confidence >= self.confidence_threshold:
                        entities.append(entity)
                
                current_entity_type = None
                current_words = []
                current_word_indices = set()
                current_boxes = []
                current_raw_boxes = []
                current_confs = []
                last_word_idx = None
        
        # Don't forget last entity
        if current_entity_type and current_words:
            entity = self._create_entity(
                current_entity_type,
                current_words,
                current_boxes,
                current_raw_boxes,
                current_confs,
                page.page_num
            )
            if entity.confidence >= self.confidence_threshold:
                entities.append(entity)
        
        return entities
    
    def _create_entity(
        self,
        entity_type: str,
        words: List[str],
        boxes: List[List[int]],
        raw_boxes: List[List[int]],
        confidences: List[float],
        page_num: int
    ) -> ExtractedEntity:
        """Create an ExtractedEntity from collected tokens."""
        # Merge bounding boxes
        merged_box = [
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes)
        ]
        
        merged_raw_box = None
        if raw_boxes:
            merged_raw_box = [
                min(b[0] for b in raw_boxes),
                min(b[1] for b in raw_boxes),
                max(b[2] for b in raw_boxes),
                max(b[3] for b in raw_boxes)
            ]
        
        return ExtractedEntity(
            entity_type=entity_type,
            text=" ".join(words),
            confidence=sum(confidences) / len(confidences),
            page=page_num,
            bbox=merged_box,
            bbox_pixels=merged_raw_box
        )
    
    def _group_by_uds_measure(
        self, 
        entities: List[ExtractedEntity]
    ) -> Dict[str, List[ExtractedEntity]]:
        """Group extracted entities by UDS measure type."""
        grouped = {}
        
        for measure_name, entity_types in self.uds_measures.items():
            matching = [
                e for e in entities 
                if e.entity_type in entity_types
            ]
            if matching:
                grouped[measure_name] = matching
        
        return grouped
    
    def batch_extract(
        self,
        input_dir: str,
        output_dir: str,
        extensions: List[str] = [".pdf"]
    ) -> List[ExtractionResult]:
        """Process all documents in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = []
        for ext in extensions:
            files.extend(input_dir.glob(f"*{ext}"))
        
        results = []
        for file_path in tqdm(files, desc="Processing documents"):
            try:
                result = self.extract_from_pdf(str(file_path))
                
                # Save individual result
                output_file = output_dir / f"{file_path.stem}_extracted.json"
                result.to_json(str(output_file))
                
                results.append(result)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Save summary
        summary = {
            "total_documents": len(results),
            "total_entities": sum(len(r.entities) for r in results),
            "documents": [r.source_file for r in results]
        }
        with open(output_dir / "extraction_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract UDS metrics from documents")
    parser.add_argument("input", help="Input PDF file or directory")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--output", default="./extractions", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--labels", default="src.labels_crc_triage", help="Labels module")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for inference")
    
    args = parser.parse_args()
    
    extractor = UDSExtractor(
        model_path=args.model,
        confidence_threshold=args.threshold,
        labels_module=args.labels,
        device=args.device
    )
    
    input_path = Path(args.input)
    if input_path.is_file():
        result = extractor.extract_from_pdf(str(input_path))
        print(f"\nExtracted {len(result.entities)} entities:")
        for entity in result.entities:
            print(f"  {entity.entity_type}: {entity.text} ({entity.confidence:.2f})")
        
        # Show UDS metrics grouping
        if result.uds_metrics:
            print(f"\nUDS Metrics Summary:")
            for measure, ents in result.uds_metrics.items():
                print(f"  {measure}:")
                for e in ents:
                    print(f"    - {e.entity_type}: {e.text}")
    else:
        results = extractor.batch_extract(str(input_path), args.output)
        print(f"\nProcessed {len(results)} documents")