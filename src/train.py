"""Training script for UDS LayoutLMv3 model."""

import os
from pathlib import Path
from typing import Dict, Optional
import yaml

import torch
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

from .labels import LABEL2ID, ID2LABEL, NUM_LABELS
from .dataset import UDSDataCollator


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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
                true_label.append(ID2LABEL[l])
                true_pred.append(ID2LABEL[p])
        
        true_labels.append(true_label)
        true_predictions.append(true_pred)
    
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


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
    
    print("=" * 50)
    print("UDS Metrics Extraction - Training")
    print("=" * 50)
    
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
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        warmup_ratio=train_config["warmup_ratio"],
        fp16=train_config["fp16"] and torch.cuda.is_available(),
        eval_strategy=train_config["eval_strategy"],
        save_strategy=train_config["save_strategy"],
        logging_steps=train_config["logging_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        report_to="none",  # Disable wandb/tensorboard by default
        dataloader_num_workers=0,  # Windows compatibility
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
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