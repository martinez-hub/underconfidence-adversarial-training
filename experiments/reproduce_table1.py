"""
Reproduce Table 1 from the UAT paper.

This script trains all 4 methods and evaluates them on multiple attacks
to reproduce the main comparison table from the paper:

Methods:
1. Vanilla (clean training)
2. PGD-AT (PGD adversarial training, 10 steps)
3. UAT-ConfSmooth (ConfSmooth adversarial training, 5 steps)
4. UAT-ClassAmbiguity (Class-Pair Ambiguity adversarial training, 10 steps)

Evaluation attacks:
- Clean (no attack)
- PGD (L-inf, epsilon=8/255)
- ConfSmooth (underconfidence attack)
- ClassAmbiguity (underconfidence attack)

Usage:
    python experiments/reproduce_table1.py --epochs 200 --device cuda
    python experiments/reproduce_table1.py --quick-test  # 1 epoch for testing
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.attacks.class_ambiguity import ClassPairAmbiguityAttack
from src.attacks.confsmooth import ConfSmoothAttack
from src.attacks.pgd import PGDAttack
from src.data.cifar10 import get_cifar10_loaders
from src.models.resnet import get_resnet18_cifar10
from src.training.trainer import Trainer
from src.utils.calibration import compute_calibration_metrics
from src.utils.checkpoints import load_checkpoint, save_checkpoint
from src.utils.config import setup_seed
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def train_model(
    method_name,
    attack_type,
    train_loader,
    val_loader,
    device,
    epochs=200,
    output_dir="checkpoints/paper_reproduction",
):
    """
    Train a single model with specified attack type.

    Args:
        method_name: Name for this method (vanilla, pgd_at, uat_confsmooth, uat_ambiguity)
        attack_type: Attack type for training (vanilla, pgd, confsmooth, class_ambiguity)
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs: Number of training epochs
        output_dir: Directory to save checkpoints

    Returns:
        Path to saved checkpoint
    """
    logger.info("="*60)
    logger.info(f"Training: {method_name}")
    logger.info("="*60)

    # Initialize model
    model = get_resnet18_cifar10()
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 150],
        gamma=0.1,
    )

    # Create trainer
    # Build a minimal config dict for trainer
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        'training': {'attack_type': attack_type},
        'attack': {'epsilon': 8/255, 'alpha': 2/255},
        'uat': {'target_class_boost': 0.01, 'pair_mode': 'top2'},
        'data': {'num_classes': 10},
        'optim': {'epochs': epochs, 'milestones': [100, 150], 'gamma': 0.1},
        'logging': {
            'log_every': 50,
            'save_every': 50,
            'output_dir': f"{output_dir}/{method_name}",
        },
    })

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        cfg=cfg,
    )

    # Train
    trainer.fit()

    # Save final checkpoint
    checkpoint_path = Path(output_dir) / method_name / f"final_epoch{epochs}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, optimizer, epochs, str(checkpoint_path))

    logger.info(f"Training complete: {method_name}")
    logger.info(f"Checkpoint saved: {checkpoint_path}")

    return checkpoint_path


def evaluate_model_all_attacks(model, test_loader, device, epsilon=8/255, alpha=2/255):
    """
    Evaluate a model on clean and all adversarial attacks.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        epsilon: Attack epsilon
        alpha: Attack alpha

    Returns:
        Dictionary with metrics for each attack type
    """
    import torch.nn.functional as F
    from tqdm import tqdm

    model.eval()

    # Initialize attacks
    pgd = PGDAttack(model, epsilon=epsilon, alpha=alpha, num_steps=20)
    confsmooth = ConfSmoothAttack(
        model, epsilon=epsilon, alpha=alpha, num_steps=20, num_classes=10
    )
    ambiguity = ClassPairAmbiguityAttack(
        model, epsilon=epsilon, alpha=alpha, num_steps=20, target_pair_mode="top2"
    )

    attacks = {
        'Clean': None,
        'PGD': pgd,
        'ConfSmooth': confsmooth,
        'ClassAmbiguity': ambiguity,
    }

    results = {}

    for attack_name, attack in attacks.items():
        logger.info(f"Evaluating on {attack_name}...")

        total = 0
        correct = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for x, y in tqdm(test_loader, desc=attack_name):
                x, y = x.to(device), y.to(device)

                # Generate adversarial examples (or use clean)
                if attack is not None:
                    x = attack.generate(x, y)

                # Evaluate
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                pred = logits.argmax(1)

                correct += pred.eq(y).sum().item()
                total += y.size(0)

                all_probs.append(probs)
                all_labels.append(y)

        # Compute metrics
        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        calib = compute_calibration_metrics(all_probs, all_labels, n_bins=15)

        results[attack_name] = {
            'accuracy': 100.0 * correct / total,
            'ece': calib['ece'],
            'mce': calib['mce'],
            'brier': calib['brier'],
        }

        logger.info(f"{attack_name}: Acc={results[attack_name]['accuracy']:.2f}%, "
                   f"ECE={results[attack_name]['ece']:.4f}")

    return results


def generate_table(all_results):
    """
    Generate comparison table from results.

    Args:
        all_results: Dictionary of results for each method

    Returns:
        Pandas DataFrame with formatted table
    """
    # Extract data for table
    rows = []
    for method_name, results in all_results.items():
        row = {'Method': method_name}
        for attack_name, metrics in results.items():
            row[f"{attack_name}_Acc"] = f"{metrics['accuracy']:.2f}%"
            row[f"{attack_name}_ECE"] = f"{metrics['ece']:.4f}"
        rows.append(row)

    df = pd.DataFrame(rows)

    return df


def main(args):
    """Main function to reproduce Table 1."""
    logger.info("="*60)
    logger.info("REPRODUCING TABLE 1 FROM UAT PAPER")
    logger.info("="*60)

    # Setup
    setup_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Determine epochs
    if args.quick_test:
        epochs = 1
        logger.info("Quick test mode: 1 epoch only")
    else:
        epochs = args.epochs
        logger.info(f"Training for {epochs} epochs")

    # Load data
    logger.info("Loading CIFAR-10...")
    train_loader, val_loader = get_cifar10_loaders(
        batch_size=128,
        num_workers=args.num_workers,
        augment=True,
    )
    _, test_loader = get_cifar10_loaders(
        batch_size=128,
        num_workers=args.num_workers,
        augment=False,
    )

    # Define methods to train
    methods = [
        ('vanilla', 'vanilla'),
        ('pgd_at', 'pgd'),
        ('uat_confsmooth', 'confsmooth'),
        ('uat_ambiguity', 'class_ambiguity'),
    ]

    all_results = {}

    # Train and evaluate each method
    for method_name, attack_type in methods:
        # Check if checkpoint exists (skip training if so)
        checkpoint_path = Path(args.output_dir) / method_name / f"final_epoch{epochs}.pt"

        if args.skip_training and checkpoint_path.exists():
            logger.info(f"Loading existing checkpoint: {checkpoint_path}")
            model = get_resnet18_cifar10().to(device)
            load_checkpoint(str(checkpoint_path), model, device=device)
        else:
            # Train model
            checkpoint_path = train_model(
                method_name=method_name,
                attack_type=attack_type,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=epochs,
                output_dir=args.output_dir,
            )

            # Load trained model
            model = get_resnet18_cifar10().to(device)
            load_checkpoint(str(checkpoint_path), model, device=device)

        # Evaluate on all attacks
        results = evaluate_model_all_attacks(model, test_loader, device)
        all_results[method_name] = results

    # Generate and print table
    logger.info("\n" + "="*60)
    logger.info("TABLE 1: COMPARISON OF ALL METHODS")
    logger.info("="*60)

    df = generate_table(all_results)
    logger.info(f"\n{df.to_string(index=False)}\n")

    # Save to CSV
    csv_path = Path(args.output_dir) / "table1_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to: {csv_path}")

    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*60)

    for method_name, results in all_results.items():
        clean_acc = results['Clean']['accuracy']
        pgd_acc = results['PGD']['accuracy']
        conf_acc = results['ConfSmooth']['accuracy']
        amb_acc = results['ClassAmbiguity']['accuracy']

        logger.info(f"\n{method_name}:")
        logger.info(f"  Clean Accuracy:         {clean_acc:.2f}%")
        logger.info(f"  PGD Robust Accuracy:    {pgd_acc:.2f}%")
        logger.info(f"  ConfSmooth Rob. Acc:    {conf_acc:.2f}%")
        logger.info(f"  ClassAmbiguity Rob. Acc: {amb_acc:.2f}%")

    logger.info("="*60)
    logger.info("Table 1 reproduction complete!")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Table 1 from UAT paper")
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs (default: 200)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu, default: cuda)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers (default: 4)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/paper_reproduction",
        help="Output directory for checkpoints (default: checkpoints/paper_reproduction)"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode: 1 epoch only"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only evaluate existing checkpoints"
    )

    args = parser.parse_args()
    main(args)
