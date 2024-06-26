import torch, torch.nn as nn, torch.utils.data
from argparse import ArgumentParser
import logging
import torch
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup
import whisper

import lightning as L

class LitWhisper(L.LightningModule):
    def __init__(self, 
        model: str = "small",
        lr:float = 1e-4,
        weight_decay: float = 0.00,
        warmup_steps: int = 500,
        optimizer_name: str = "adam",
        labeled_w: float = 1.0,
        pseudo_labeled_w: float = 0.1
        *args, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = whisper.load_model(self.hparams.model, device=self.device)
        del self.model.alignment_heads

    def model_step(self, batch, batch_idx, name, labeled_w, pseudo_labeled_w):
        x, y_in, y_out, is_labeled = batch
        x, y_in, y_out = x.to(self.device), y_in.to(self.device), y_out.to(self.device)
        audio_features = self.model.embed_audio(x)
        logits = self.model.logits(y_in, audio_features=audio_features)
        
        labeled_idx = is_labeled == 1
        pseudo_labeled_idx = is_labeled == 0
        
        labeled_loss = F.cross_entropy(logits[labeled_idx].transpose(1, 2), y_out[labeled_idx], ignore_index=-100) if labeled_idx.any() else 0
        pseudo_labeled_loss = F.cross_entropy(logits[pseudo_labeled_idx].transpose(1, 2), y_out[pseudo_labeled_idx], ignore_index=-100) if pseudo_labeled_idx.any() else 0

        loss = (labeled_w * labeled_loss) + (pseudo_labeled_w * pseudo_labeled_loss)
		
        self.log(f"{name}_loss", loss, prog_bar=True, sync_dist=True)                
        return loss        

    def training_step(self, batch, batch_idx):
        loss = self.model_step(batch, batch_idx, "train")
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.get_last_lr()[0], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, "eval")

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)
        # update learning rate
        self.lr_schedulers().step()


    def configure_optimizers(self):
        if self.hparams.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)    
        elif self.hparams.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)                        
        elif self.hparams.optimizer_name == "adam8bit":
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            raise Exception(f"Unknown optimizer ({self.hparams.optimizer_name})")
        logging.info(f"Estimated stepping batches: {self.trainer.estimated_stepping_batches}")
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [scheduler]



