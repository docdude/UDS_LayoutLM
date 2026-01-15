"""Training script for UDS LayoutLMv3 model."""

import os
import importlib
from pathlib import Path
from typing import Dict, Optional
from collections import Counter
import yaml

import torch
import torch.nn as nn
import numpy as np
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_from_disk
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score
)

from .dataset import UDSDataCollator


class WeightedTrainer(Trainer):
    """Custom Trainer with class-weighted loss for imbalanced NER."""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(dataset, num_labels: int):
    """Compute inverse frequency weights for each class."""
    all_labels = []
    for ex in dataset:
        all_labels.extend(ex['labels'])
    
    label_counts = Counter(all_labels)
    total = sum(label_counts.values())
    
    # Compute weights: higher weight for rare classes
    weights = torch.ones(num_labels)
    for label_id, count in label_counts.items():
        if label_id >= 0:  # Skip -100 (padding)
            # Inverse frequency, capped to avoid extreme weights
            weights[label_id] = min(total / (count * num_labels), 50.0)
    
    # Give extra weight to non-O classes (multiply by 10)
    for i in range(1, num_labels):
        weights[i] *= 10.0
    
    print(f"Class weights: O={weights[0]:.2f}, avg non-O={weights[1:].mean():.2f}")
    return weights


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_compute_metrics(id2label):
    """Create compute_metrics function with the given id2label mapping."""
    def compute_metrics(eval_pred):
        """Compute seqeval metrics for token classification."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        true_labels = []
        true_predictions = []
        
        for prediction, label in zip(predictions, labels):
            true_label = []
            true_pred = []
            
            for p, l in zip(prediction, label):
                if l != -100:  # Ignore padding and special tokens
                    true_label.append(id2label[l])
                    true_pred.append(id2label[p])
            
            true_labels.append(true_label)
            true_predictions.append(true_pred)
        
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
    return compute_metrics


def train(
    config_path: str = "config.yaml",
    resume_from_checkpoint: Optional[str] = None
):
    """Main training function."""
    
    # Load config
    config = load_config(config_path)
    model_config = config["model"]
    train_config = config["training"]
    data_config = config["data"]
    
    # Load labels - support configurable label modules
    labels_module_name = config.get("labels", {}).get("module", "src.labels")
    print(f"Loading labels from: {labels_module_name}")
    labels_module = importlib.import_module(labels_module_name)
    LABEL2ID = labels_module.LABEL2ID
    ID2LABEL = labels_module.ID2LABEL
    NUM_LABELS = labels_module.NUM_LABELS
    
    print("=" * 50)
    print("UDS Metrics Extraction - Training")
    print("=" * 50)
    print(f"Using {NUM_LABELS} labels from {labels_module_name}")
    
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load processor
    print(f"\nLoading processor from {model_config['base_model']}...")
    processor = LayoutLMv3Processor.from_pretrained(
        model_config["base_model"],
        apply_ocr=False  # We provide our own OCR
    )
    
    # Load model
    print(f"Loading model from {model_config['base_model']}...")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_config["base_model"],
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    
    # Load datasets
    print(f"\nLoading dataset from {data_config['dataset']}...")
    dataset = load_from_disk(data_config["dataset"])
    
    print(f"  Train examples: {len(dataset['train'])}")
    print(f"  Validation examples: {len(dataset['validation'])}")
    
    # Data collator
    data_collator = UDSDataCollator(
        processor=processor,
        max_length=model_config["max_length"]
    )
    
    # Output directory
    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_config["epochs"],
        per_device_train_batch_size=train_config["batch_size"],
        per_device_eval_batch_size=train_config["batch_size"],
        learning_rate=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        warmup_ratio=float(train_config["warmup_ratio"]),
        fp16=train_config["fp16"] and torch.cuda.is_available(),
        eval_strategy=train_config["eval_strategy"],
        save_strategy=train_config["save_strategy"],
        logging_steps=train_config["logging_steps"],
        save_total_limit=3,
        report_to="none",  # Disable wandb/tensorboard by default
        dataloader_num_workers=0,  # Windows compatibility
        remove_unused_columns=False,  # Keep all columns for custom collator
    )
    
    # Compute class weights to handle imbalance
    print("\nComputing class weights for imbalanced data...")
    class_weights = compute_class_weights(dataset["train"], NUM_LABELS)
    
    # Create compute_metrics with our labels
    compute_metrics = make_compute_metrics(ID2LABEL)
    
    # Trainer with weighted loss
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\nStarting training...")
    print("-" * 50)
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save final model
    final_model_path = output_dir / "final_model"
    print(f"\nSaving final model to {final_model_path}...")
    trainer.save_model(str(final_model_path))
    processor.save_pretrained(str(final_model_path))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(dataset["test"])
    print(f"Test Results: {test_results}")
    
    # Save results
    results_file = output_dir / "training_results.yaml"
    with open(results_file, "w") as f:
        yaml.dump({
            "test_results": test_results,
            "config": config
        }, f)
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Model saved to: {final_model_path}")
    print("=" * 50)
    
    return trainer, test_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train UDS LayoutLMv3 model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Resume from checkpoint"
    )
    
    args = parser.parse_args()
    train(config_path=args.config, resume_from_checkpoint=args.resume)