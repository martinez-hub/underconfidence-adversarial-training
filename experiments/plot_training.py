"""Plot training curves from saved checkpoints."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch


def load_history_from_checkpoint(checkpoint_path: str) -> Dict[str, List[float]]:
    """
    Load training history from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary containing training history

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        KeyError: If checkpoint doesn't contain history
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "history" not in checkpoint:
        raise KeyError(
            f"Checkpoint {checkpoint_path} does not contain training history. "
            "Make sure the model was trained with history tracking enabled."
        )

    return checkpoint["history"]


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot training curves (loss and accuracy).

    Args:
        history: Training history dictionary
        save_path: Optional path to save plot
        show: Whether to display plot interactively
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss curves
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot accuracy curves
    ax2.plot(epochs, history["train_acc_clean"], "b-", label="Train Acc (Clean)", linewidth=2)
    if "train_acc_train" in history:
        ax2.plot(epochs, history["train_acc_train"], "g--", label="Train Acc (Train)", linewidth=2, alpha=0.7)
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_learning_rate(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot learning rate schedule.

    Args:
        history: Training history dictionary
        save_path: Optional path to save plot
        show: Whether to display plot interactively
    """
    if "learning_rate" not in history:
        print("Warning: No learning rate history found")
        return

    epochs = range(1, len(history["learning_rate"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["learning_rate"], "b-", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.title("Learning Rate Schedule", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")  # Log scale for better visualization

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def compare_training_runs(
    checkpoint_paths: List[str],
    labels: List[str],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Compare training curves from multiple runs.

    Args:
        checkpoint_paths: List of checkpoint paths
        labels: Labels for each run
        save_path: Optional path to save plot
        show: Whether to display plot interactively
    """
    if len(checkpoint_paths) != len(labels):
        raise ValueError("Number of checkpoint paths must match number of labels")

    # Load all histories
    histories = []
    for path in checkpoint_paths:
        try:
            history = load_history_from_checkpoint(path)
            histories.append(history)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
            continue

    if not histories:
        print("Error: No valid histories loaded")
        return

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot validation accuracy comparison
    colors = plt.cm.tab10(range(len(histories)))
    for i, (history, label) in enumerate(zip(histories, labels)):
        epochs = range(1, len(history["val_acc"]) + 1)
        ax1.plot(epochs, history["val_acc"], color=colors[i], label=label, linewidth=2)

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax1.set_title("Validation Accuracy Comparison", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot validation loss comparison
    for i, (history, label) in enumerate(zip(histories, labels)):
        epochs = range(1, len(history["val_loss"]) + 1)
        ax2.plot(epochs, history["val_loss"], color=colors[i], label=label, linewidth=2)

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Validation Loss", fontsize=12)
    ax2.set_title("Validation Loss Comparison", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def print_training_summary(history: Dict[str, List[float]]):
    """
    Print summary statistics from training history.

    Args:
        history: Training history dictionary
    """
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    num_epochs = len(history["train_loss"])
    print(f"Total Epochs: {num_epochs}")

    print(f"\nFinal Metrics (Epoch {num_epochs}):")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Train Acc (Clean): {history['train_acc_clean'][-1]:.2f}%")
    if "train_acc_train" in history:
        print(f"  Train Acc (Train): {history['train_acc_train'][-1]:.2f}%")
    print(f"  Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"  Val Acc: {history['val_acc'][-1]:.2f}%")

    # Best validation accuracy
    best_val_acc = max(history["val_acc"])
    best_epoch = history["val_acc"].index(best_val_acc) + 1
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")

    # Lowest validation loss
    best_val_loss = min(history["val_loss"])
    best_loss_epoch = history["val_loss"].index(best_val_loss) + 1
    print(f"Lowest Validation Loss: {best_val_loss:.4f} (Epoch {best_loss_epoch})")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (or multiple paths separated by commas for comparison)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Labels for comparison (comma-separated, required if multiple checkpoints)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for plot (default: show interactively)",
    )
    parser.add_argument(
        "--lr-plot",
        action="store_true",
        help="Also plot learning rate schedule",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't show plot interactively (only save)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print training summary statistics",
    )

    args = parser.parse_args()

    # Parse checkpoint paths
    checkpoint_paths = [p.strip() for p in args.checkpoint.split(",")]

    # Handle comparison mode (multiple checkpoints)
    if len(checkpoint_paths) > 1:
        if not args.labels:
            # Use filenames as labels by default
            labels = [Path(p).stem for p in checkpoint_paths]
        else:
            labels = [l.strip() for l in args.labels.split(",")]

        compare_training_runs(
            checkpoint_paths,
            labels,
            save_path=args.output,
            show=not args.no_show,
        )

    # Handle single checkpoint mode
    else:
        checkpoint_path = checkpoint_paths[0]

        # Load history
        history = load_history_from_checkpoint(checkpoint_path)

        # Print summary if requested
        if args.summary:
            print_training_summary(history)

        # Plot training curves
        plot_training_curves(
            history,
            save_path=args.output,
            show=not args.no_show,
        )

        # Plot learning rate if requested
        if args.lr_plot:
            lr_output = None
            if args.output:
                lr_output = Path(args.output).with_name(
                    Path(args.output).stem + "_lr" + Path(args.output).suffix
                )
            plot_learning_rate(
                history,
                save_path=lr_output,
                show=not args.no_show,
            )


if __name__ == "__main__":
    main()
