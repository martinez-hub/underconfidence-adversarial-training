"""Evaluation script for trained models."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.attacks.class_ambiguity import ClassPairAmbiguityAttack
from src.attacks.confsmooth import ConfSmoothAttack
from src.attacks.pgd import PGDAttack
from src.data.cifar10 import get_cifar10_loaders
from src.models.resnet import get_resnet18_cifar10
from src.utils.calibration import compute_calibration_metrics, plot_reliability_diagram
from src.utils.checkpoints import load_checkpoint
from src.utils.config import load_config, setup_device
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate_model(model, test_loader, device, attacks=None, attack_names=None):
    """
    Evaluate model on clean and adversarial test sets.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        attacks: List of attack objects (optional)
        attack_names: List of attack names (optional)

    Returns:
        Dictionary of metrics for each attack type
    """
    model.eval()
    results = {}

    # Clean evaluation
    logger.info("Evaluating on clean images...")
    clean_acc = 0.0
    clean_conf = []
    total = 0

    # Collect all probabilities and labels for calibration metrics
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Clean"):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            probs = F.softmax(logits, dim=1)

            pred = logits.argmax(1)
            clean_acc += pred.eq(y).sum().item()

            # Collect for calibration metrics
            all_probs.append(probs)
            all_labels.append(y)

            # Confidence on correct predictions
            correct_mask = pred.eq(y)
            if correct_mask.sum() > 0:
                correct_probs = probs[correct_mask]
                correct_conf = correct_probs.max(dim=1)[0]
                clean_conf.extend(correct_conf.cpu().tolist())

            total += y.size(0)

    # Compute calibration metrics
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    calib_metrics = compute_calibration_metrics(all_probs, all_labels, n_bins=15)

    results["clean"] = {
        "accuracy": 100.0 * clean_acc / total,
        "confidence_mean": sum(clean_conf) / len(clean_conf) if clean_conf else 0.0,
        "ece": calib_metrics['ece'],
        "mce": calib_metrics['mce'],
        "brier": calib_metrics['brier'],
    }

    logger.info(f"Clean accuracy: {results['clean']['accuracy']:.2f}%")
    logger.info(f"Clean confidence: {results['clean']['confidence_mean']:.4f}")
    logger.info(f"Clean ECE: {results['clean']['ece']:.4f}")
    logger.info(f"Clean MCE: {results['clean']['mce']:.4f}")
    logger.info(f"Clean Brier: {results['clean']['brier']:.4f}")

    # Adversarial evaluation
    if attacks:
        for attack, attack_name in zip(attacks, attack_names):
            logger.info(f"\nEvaluating on {attack_name} adversarial images...")
            adv_acc = 0.0
            adv_conf = []

            # Collect for calibration metrics
            all_adv_probs = []
            all_adv_labels = []

            for x, y in tqdm(test_loader, desc=attack_name):
                x, y = x.to(device), y.to(device)

                # Generate adversarial examples
                x_adv = attack.generate(x, y)

                with torch.no_grad():
                    logits_adv = model(x_adv)
                    probs_adv = F.softmax(logits_adv, dim=1)

                    pred_adv = logits_adv.argmax(1)
                    adv_acc += pred_adv.eq(y).sum().item()

                    # Collect for calibration metrics
                    all_adv_probs.append(probs_adv)
                    all_adv_labels.append(y)

                    # Confidence on correct predictions
                    correct_mask = pred_adv.eq(y)
                    if correct_mask.sum() > 0:
                        correct_probs = probs_adv[correct_mask]
                        correct_conf = correct_probs.max(dim=1)[0]
                        adv_conf.extend(correct_conf.cpu().tolist())

            # Compute calibration metrics
            all_adv_probs = torch.cat(all_adv_probs, dim=0)
            all_adv_labels = torch.cat(all_adv_labels, dim=0)
            adv_calib = compute_calibration_metrics(all_adv_probs, all_adv_labels, n_bins=15)

            results[attack_name] = {
                "accuracy": 100.0 * adv_acc / total,
                "confidence_mean": sum(adv_conf) / len(adv_conf) if adv_conf else 0.0,
                "ece": adv_calib['ece'],
                "mce": adv_calib['mce'],
                "brier": adv_calib['brier'],
            }

            logger.info(f"{attack_name} accuracy: {results[attack_name]['accuracy']:.2f}%")
            logger.info(f"{attack_name} confidence: {results[attack_name]['confidence_mean']:.4f}")
            logger.info(f"{attack_name} ECE: {results[attack_name]['ece']:.4f}")
            logger.info(f"{attack_name} MCE: {results[attack_name]['mce']:.4f}")
            logger.info(f"{attack_name} Brier: {results[attack_name]['brier']:.4f}")

    return results


def main(checkpoint_path: str, config_path: str, eval_attacks: bool = True):
    """
    Main evaluation function.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        eval_attacks: Whether to evaluate on adversarial examples
    """
    cfg = load_config(config_path)
    device = setup_device(cfg.meta.device)

    logger.info("="*60)
    logger.info("Evaluation Configuration")
    logger.info("="*60)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Device: {device}")
    logger.info("="*60)

    # Data
    logger.info("Loading CIFAR-10 test set...")
    _, test_loader = get_cifar10_loaders(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Model
    logger.info("Loading model...")
    model = get_resnet18_cifar10()
    model = model.to(device)

    # Load checkpoint
    epoch = load_checkpoint(checkpoint_path, model, device=device)
    logger.info(f"Loaded checkpoint from epoch {epoch}")

    # Attacks
    attacks = None
    attack_names = None

    if eval_attacks:
        logger.info("Initializing attacks for evaluation...")
        pgd = PGDAttack(
            model,
            epsilon=cfg.attack.epsilon,
            alpha=cfg.attack.alpha,
            num_steps=20
        )
        ambiguity = ClassPairAmbiguityAttack(
            model,
            epsilon=cfg.attack.epsilon,
            alpha=cfg.attack.alpha,
            num_steps=20,
            target_pair_mode="top2"
        )
        confsmooth = ConfSmoothAttack(
            model,
            epsilon=cfg.attack.epsilon,
            alpha=cfg.attack.alpha,
            num_steps=20,
            num_classes=10
        )

        attacks = [pgd, ambiguity, confsmooth]
        attack_names = ["PGD", "ClassAmbiguity", "ConfSmooth"]
        logger.info(f"Initialized {len(attacks)} attacks: {', '.join(attack_names)}")

    # Evaluate
    logger.info("="*60)
    logger.info("Starting evaluation...")
    logger.info("="*60)
    results = evaluate_model(model, test_loader, device, attacks, attack_names)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("="*60)

    # Print table header
    logger.info(f"\n{'Attack':<15} {'Accuracy':<12} {'Confidence':<12} {'ECE':<10} {'MCE':<10} {'Brier':<10}")
    logger.info("-" * 69)

    # Print metrics for each attack
    for attack_name, metrics in results.items():
        logger.info(
            f"{attack_name:<15} "
            f"{metrics['accuracy']:>10.2f}% "
            f"{metrics['confidence_mean']:>11.4f} "
            f"{metrics.get('ece', 0):>9.4f} "
            f"{metrics.get('mce', 0):>9.4f} "
            f"{metrics.get('brier', 0):>9.4f}"
        )

    logger.info("="*60)

    # Print detailed metrics
    logger.info("\nDetailed Metrics:")
    for attack_name, metrics in results.items():
        logger.info(f"\n{attack_name}:")
        for metric_name, value in metrics.items():
            if "accuracy" in metric_name:
                logger.info(f"  {metric_name}: {value:.2f}%")
            else:
                logger.info(f"  {metric_name}: {value:.4f}")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained UAT models")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "--no-attacks",
        action="store_true",
        help="Skip adversarial evaluation (only evaluate on clean images)"
    )
    args = parser.parse_args()

    main(args.checkpoint, args.config, eval_attacks=not args.no_attacks)
