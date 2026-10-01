from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import BertConfig, BertForTokenClassification, TrainerCallback, TrainingArguments

from train.config import NER_LABELS
from train.training import CampaignBudgetCallback, FeatureDataset, WeightedNERTrainer, WeightedTokenCollator


class StopAfterTwoSteps(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 2:
            control.should_training_stop = True
        return control


def run(mode: str, root: Path) -> None:
    torch.manual_seed(42)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = Path("/source/model")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    labels = {index: name for index, name in enumerate(NER_LABELS)}
    config = BertConfig(
        vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1,
        num_attention_heads=4, intermediate_size=64, max_position_embeddings=32,
        num_labels=len(NER_LABELS), id2label=labels,
        label2id={name: index for index, name in labels.items()},
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )
    model = BertForTokenClassification(config)
    rows = []
    for row_index in range(8):
        rows.append({
            "input_ids": [tokenizer.cls_token_id, 100 + row_index, 200 + row_index, tokenizer.sep_token_id],
            "attention_mask": [1, 1, 1, 1],
            "token_type_ids": [0, 0, 0, 0],
            "labels": [-100, row_index % len(NER_LABELS), 0, -100],
            "loss_weights": [0.0, 1.0, 1.0, 0.0],
            "auxiliary_labels": [-100, -100, -100, -100],
            "auxiliary_weights": [0.0, 0.0, 0.0, 0.0],
        })
    output_dir = root / ("trainer_continuous" if mode == "continuous" else "trainer_interrupt")
    args = TrainingArguments(
        output_dir=str(output_dir), per_device_train_batch_size=1,
        gradient_accumulation_steps=1, learning_rate=1e-3, weight_decay=0.01,
        max_steps=4, save_strategy="steps", save_steps=2, save_total_limit=2,
        logging_strategy="no", evaluation_strategy="no", fp16=True,
        gradient_checkpointing=False, seed=42, data_seed=42, full_determinism=True,
        report_to="none", disable_tqdm=True, remove_unused_columns=False,
        save_safetensors=True, save_only_model=False,
        restore_callback_states_from_checkpoint=True,
    )
    callbacks = [CampaignBudgetCallback(root, time.monotonic(), 0.0, "cuda_resume_test", 0.0)]
    if mode == "interrupt":
        callbacks.append(StopAfterTwoSteps())
    trainer = WeightedNERTrainer(
        model=model, args=args, train_dataset=FeatureDataset(rows),
        data_collator=WeightedTokenCollator(tokenizer), callbacks=callbacks,
        auxiliary_loss_weight=0.25,
    )
    checkpoint = root / "trainer_interrupt" / "checkpoint-2"
    if mode == "resume":
        trainer.train(resume_from_checkpoint=str(checkpoint))
    else:
        trainer.train()
    state = {
        "global_step": trainer.state.global_step,
        "learning_rates": trainer.lr_scheduler.get_last_lr(),
        "scaler": trainer.accelerator.scaler.state_dict(),
    }
    torch.save({key: value.detach().cpu() for key, value in trainer.model.state_dict().items()}, root / f"{mode}.weights.pt")
    torch.save(trainer.optimizer.state_dict(), root / f"{mode}.optimizer.pt")
    torch.save(trainer.lr_scheduler.state_dict(), root / f"{mode}.scheduler.pt")
    (root / f"{mode}.json").write_text(json.dumps(state, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    run(sys.argv[1], Path(sys.argv[2]))
