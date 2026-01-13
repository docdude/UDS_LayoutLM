"""Quick test of the training pipeline - runs 5 steps only."""
import os
import sys
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
sys.path.insert(0, '.')

from datasets import load_from_disk
from transformers import (
    LayoutLMv3ForTokenClassification, 
    LayoutLMv3Processor, 
    TrainingArguments, 
    Trainer
)
from src.labels import LABEL2ID, ID2LABEL, NUM_LABELS
from src.dataset import UDSDataCollator

print('Loading dataset and model...')
dataset = load_from_disk('data/dataset')
print(f'  Train: {len(dataset["train"])} examples')

processor = LayoutLMv3Processor.from_pretrained('microsoft/layoutlmv3-base', apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(
    'microsoft/layoutlmv3-base', 
    num_labels=NUM_LABELS, 
    id2label=ID2LABEL, 
    label2id=LABEL2ID
)
print(f'  Model loaded with {NUM_LABELS} labels')

data_collator = UDSDataCollator(processor=processor, max_length=512)

training_args = TrainingArguments(
    output_dir='./outputs/test_run',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    logging_steps=1,
    eval_strategy='no',
    save_strategy='no',
    report_to='none',
    dataloader_num_workers=0,
    use_cpu=True,
    max_steps=5,
    remove_unused_columns=False,  # Keep all columns, let our collator handle them
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    data_collator=data_collator
)

print('Running 5 training steps...')
result = trainer.train()
print(f'\n✓ SUCCESS - Training pipeline verified!')
print(f'  Training loss: {result.training_loss:.4f}')
print(f'  Global steps: {result.global_step}')
