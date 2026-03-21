"""
logger.py - Logging history and Checkpoint management
"""

import json
import os
import torch


class TrainingLogger:
    """
    Manages saving training history and model checkpoints.
    
    Features:
    - Records loss, accuracy, metrics after each epoch
    - Saves best model (based on lowest val_loss)
    - Saves last model (to resume training)
    - Exports history to JSON file
    """

    def __init__(self, model_dir):
        """
        Args:
            model_dir: Path to the model's result directory
                       (e.g., results/model_1_scratch_no_att)
        """
        self.model_dir = model_dir
        self.checkpoint_dir = os.path.join(model_dir, "checkpoints")
        self.log_dir = os.path.join(model_dir, "logs")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
            "teacher_forcing": [],
        }
        self.best_val_loss = float("inf")
        self.best_epoch = -1

    def log_epoch(self, epoch, train_loss, val_loss, val_accuracy,
                  lr, tf_ratio, extra_metrics=None):
        """
        Records the results of one epoch.

        Args:
            epoch: Epoch number
            train_loss: Train set loss
            val_loss: Validation set loss
            val_accuracy: Validation set accuracy
            lr: Current learning rate
            tf_ratio: Current teacher forcing ratio
            extra_metrics: Dict containing additional metrics (BLEU, etc.)
        """
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_accuracy"].append(val_accuracy)
        self.history["learning_rate"].append(lr)
        self.history["teacher_forcing"].append(tf_ratio)

        if extra_metrics:
            for key, value in extra_metrics.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)

        # Print results
        print(f"  Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.4f} | "
              f"LR: {lr:.6f} | "
              f"TF: {tf_ratio:.2f}")

        # Check for best model
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            print(f"  ★ New best model! (val_loss={val_loss:.4f})")

        return is_best

    def save_checkpoint(self, model, optimizer, epoch, is_best=False):
        """
        Save checkpoint.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            is_best: True if this is the best model
        """
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

        # Always save last model
        last_path = os.path.join(self.checkpoint_dir, "last_model.pth")
        torch.save(state, last_path)

        # Save best model if needed
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(state, best_path)

    def save_history(self):
        """Save history to JSON file."""
        history_path = os.path.join(self.log_dir, "history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self, model, optimizer=None, load_best=True):
        """
        Load checkpoint to resume training or for inference.

        Args:
            model: PyTorch model
            optimizer: Optimizer (None if only inference)
            load_best: True → load best model, False → load last model

        Returns:
            epoch: The last trained epoch
        """
        filename = "best_model.pth" if load_best else "last_model.pth"
        path = os.path.join(self.checkpoint_dir, filename)

        if not os.path.exists(path):
            print(f"  No checkpoint found at {path}")
            return 0

        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.history = checkpoint.get("history", self.history)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        epoch = checkpoint["epoch"]

        print(f"  Loaded checkpoint: {filename} (epoch {epoch})")
        return epoch


class EarlyStopping:
    """
    Stop training early if val_loss doesn't improve after N epochs.
    """

    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")

    def should_stop(self, val_loss):
        """
        Returns:
            True if training should be stopped
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\n  ⚠ Early stopping! No improvement for "
                      f"{self.patience} epochs.")
                return True
            print(f"  Early stopping counter: {self.counter}/{self.patience}")
            return False
