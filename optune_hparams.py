import optuna
import pytorch_lightning as pl
from model import LitWhisper
from data import WhisperDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, EarlyStopping

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_steps = trial.suggest_int("warmup_steps", 100, 1000)
    optimizer_name = trial.suggest_categorical("optimizer_name", ["adam", "adamw"])
    labeled_w = trial.suggest_float("labeled_w", 0.1, 1.0)
    pseudo_labeled_w = trial.suggest_float("pseudo_labeled_w", 0.01, 0.5)
    
    prompt_use_rate = trial.suggest_float("prompt_use_rate", 0.0, 1.0)
    no_timestamps_rate = trial.suggest_float("no_timestamps_rate", 0.0, 1.0)
    max_prompt_length = trial.suggest_int("max_prompt_length", 100, 300)
    n_mels = trial.suggest_int("n_mels", 60, 130)
    do_spec_augment = trial.suggest_categorical("do_spec_augment", [False, True])
    
    data_module = WhisperDataModule(
        model="large-v3",
        train_json="train.json",
        dev_json="dev.json",
        batch_size=2,
        num_workers=4,
        no_timestamps_training=False,
        prompt_use_rate=prompt_use_rate,
        no_timestamps_rate=no_timestamps_rate,
        max_prompt_length=max_prompt_length,
        n_mels=n_mels,
        do_spec_augment=do_spec_augment
    )
    
    model = LitWhisper(
        model="large-v3",
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        optimizer_name=optimizer_name,
        labeled_w=labeled_w,
        pseudo_labeled_w=pseudo_labeled_w
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="eval_loss",
        save_top_k=1,
        mode="min"
    )
    
    swa_callback = StochasticWeightAveraging(swa_epoch_start=3, swa_lrs=2e-4)
    
    trainer = pl.Trainer(
        devices=2,
        accelerator = "gpu",
        accumulate_grad_batches=16,
        val_check_interval=1.0,
        max_epochs=3,
        gradient_clip_val=1.0,
        precision="16-mixed",  # for mixed precision training
        callbacks=[checkpoint_callback, swa_callback]
    )
    
    trainer.fit(model, datamodule=data_module)
    return trainer.callback_metrics["eval_loss"].item()

if __name__ == '__main__':
    study = optuna.create_study(direction="minimize", study_name="Best HyperParams for Whisper")
    study.optimize(objective, n_trials=50)
    print("Best hyperparameters: ", study.best_params)
