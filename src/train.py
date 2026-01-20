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


def compute_class_weights(dataset, num_labels: int, entity_boost: float = 5.0, o_weight: float = 1.0):
    """
    Compute balanced class weights for NER.
    
    Args:
        dataset: Training dataset
        num_labels: Number of label classes
        entity_boost: Multiplier for entity classes vs O (default 5.0)
        o_weight: Base weight for O class (default 1.0)
    
    Strategy:
        - Use sqrt of inverse frequency (less extreme than raw inverse)
        - Separate boost for B- vs I- tags (B- slightly higher)
        - Configurable entity boost to tune precision/recall tradeoff
    """
    all_labels = []
    for ex in dataset:
        all_labels.extend(ex['labels'])
    
    label_counts = Counter(all_labels)
    total = sum(v for k, v in label_counts.items() if k >= 0)
    
    # Count how many are O vs entities
    o_count = label_counts.get(0, 1)
    entity_count = total - o_count
    
    print(f"  Label distribution: O={o_count} ({o_count/total:.1%}), entities={entity_count} ({entity_count/total:.1%})")
    
    # Compute weights
    weights = torch.ones(num_labels)
    
    # O class gets base weight (can reduce to push model toward entities)
    weights[0] = o_weight
    
    # Entity classes get sqrt inverse frequency + boost
    for label_id, count in label_counts.items():
        if label_id > 0:  # Skip O and padding
            # sqrt dampens extreme weights for rare classes
            freq_weight = min((total / (count * num_labels)) ** 0.5, 10.0)
            weights[label_id] = freq_weight * entity_boost
    
    # Slight boost for B- tags (beginning of entity) vs I- tags
    # B- tags are more important for entity detection
    from src.labels_crc_triage import ID2LABEL
    for label_id in range(1, num_labels):
        if ID2LABEL.get(label_id, "").startswith("B-"):
            weights[label_id] *= 1.2  # 20% boost for B- tags
    
    # Normalize so mean weight = 1 (prevents loss scale issues)
    weights = weights / weights.mean()
    
    print(f"  Class weights: O={weights[0]:.3f}, avg B-tags={weights[1::2].mean():.3f}, avg I-tags={weights[2::2].mean():.3f}")
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
    
    # Set up device - respect config override if specified
    device_config = train_config.get("device", "auto")
    if device_config == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config
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
    model.to(device)
    
    # Load datasets
    print(f"\nLoading dataset from {data_config['dataset']}...")
    dataset = load_from_disk(data_config["dataset"])
    
    print(f"  Train examples: {len(dataset['train'])}")
    print(f"  Validation examples: {len(dataset['validation'])}")
    
    # Data collator with optional augmentation
    augment_images = train_config.get("augment_images", False)
    augment_prob = train_config.get("augment_prob", 0.5)
    print(f"\nImage augmentation: {'enabled' if augment_images else 'disabled'}")
    
    data_collator = UDSDataCollator(
        processor=processor,
        max_length=model_config["max_length"],
        augment=augment_images,
        augment_prob=augment_prob
    )
    
    # Output directory
    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for early stopping config
    early_stopping_patience = train_config.get("early_stopping_patience", None)
    load_best_model = train_config.get("load_best_model_at_end", False)
    
    # Gradient accumulation for effective larger batch sizes
    gradient_accumulation_steps = train_config.get("gradient_accumulation_steps", 1)
    effective_batch_size = train_config["batch_size"] * gradient_accumulation_steps
    print(f"Gradient accumulation: {gradient_accumulation_steps} steps (effective batch size: {effective_batch_size})")
    
    # FP16 mixed precision
    use_fp16 = train_config.get("fp16", False) and torch.cuda.is_available()
    print(f"FP16 mixed precision: {'enabled' if use_fp16 else 'disabled'}")
    
    # LR scheduler
    lr_scheduler = train_config.get("lr_scheduler_type", "linear")
    print(f"LR scheduler: {lr_scheduler}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_config["epochs"],
        per_device_train_batch_size=train_config["batch_size"],
        per_device_eval_batch_size=train_config["batch_size"],
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        warmup_ratio=float(train_config["warmup_ratio"]),
        lr_scheduler_type=lr_scheduler,
        fp16=use_fp16,
        eval_strategy=train_config["eval_strategy"],
        save_strategy=train_config["save_strategy"],
        logging_steps=train_config["logging_steps"],
        logging_dir=str(output_dir / "logs"),  # TensorBoard logs directory
        save_total_limit=3,
        report_to=train_config.get("report_to", "none"),  # TensorBoard/wandb/none
        dataloader_num_workers=0,  # Windows compatibility
        remove_unused_columns=False,  # Keep all columns for custom collator
        no_cuda=(device == "cpu"),  # Force CPU mode if specified
        load_best_model_at_end=load_best_model,
        metric_for_best_model="f1" if load_best_model else None,
        greater_is_better=True if load_best_model else None,
    )
    
    # Compute class weights to handle imbalance
    print("\nComputing class weights for imbalanced data...")
    entity_boost = train_config.get("entity_boost", 5.0)
    o_weight = train_config.get("o_weight", 1.0)
    print(f"  Using entity_boost={entity_boost}, o_weight={o_weight}")
    class_weights = compute_class_weights(dataset["train"], NUM_LABELS, entity_boost=entity_boost, o_weight=o_weight)
    
    # Create compute_metrics with our labels
    compute_metrics = make_compute_metrics(ID2LABEL)
    
    # Set up callbacks
    callbacks = []
    if early_stopping_patience:
        print(f"Early stopping enabled: patience={early_stopping_patience} epochs")
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
    
    # Trainer with weighted loss
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks if callbacks else None,
    )
    
    # Train
    print("\nStarting training...")
    print("-" * 50)
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Get best model info (if load_best_model_at_end is True)
    best_metric = None
    best_epoch = None
    if load_best_model and hasattr(trainer.state, 'best_metric'):
        best_metric = trainer.state.best_metric
        # Find which epoch had the best metric from log history
        for log in reversed(trainer.state.log_history):
            if 'eval_f1' in log and abs(log['eval_f1'] - best_metric) < 1e-6:
                best_epoch = log.get('epoch')
                break
        print(f"\nBest model loaded from epoch {best_epoch} with F1={best_metric:.4f}")
    
    # Save final model (this is the best model if load_best_model_at_end=True)
    final_model_path = output_dir / "final_model"
    print(f"\nSaving final model to {final_model_path}...")
    trainer.save_model(str(final_model_path))
    processor.save_pretrained(str(final_model_path))
    
    # Evaluate on test set (using the best model)
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(dataset["test"])
    print(f"Test Results: {test_results}")
    
    # Save results with best model info
    results_file = output_dir / "training_results.yaml"
    results_data = {
        "test_results": test_results,
        "config": config
    }
    if best_metric is not None:
        results_data["best_model"] = {
            "metric": "f1",
            "value": float(best_metric),
            "epoch": float(best_epoch) if best_epoch else None
        }
    with open(results_file, "w") as f:
        yaml.dump(results_data, f)
    
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