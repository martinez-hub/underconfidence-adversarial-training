"""
Verification script for underconfidence attacks.

This script verifies that:
1. PGD attack increases loss (standard adversarial behavior)
2. ConfSmooth attack reduces confidence while maintaining accuracy
3. ClassAmbiguity attack reduces margin while maintaining accuracy
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.attacks.class_ambiguity import ClassPairAmbiguityAttack
from src.attacks.confsmooth import ConfSmoothAttack
from src.attacks.pgd import PGDAttack
from src.data.cifar10 import get_cifar10_loaders
from src.models.resnet import get_resnet18_cifar10
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_metrics(model, x, y, device):
    """Compute accuracy, confidence, entropy, and margin."""
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)

        # Accuracy
        accuracy = pred.eq(y).float().mean().item() * 100

        # Confidence (max probability)
        max_probs = probs.max(dim=1)[0]
        mean_confidence = max_probs.mean().item()

        # Entropy (measure of uncertainty)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=1)
        mean_entropy = entropy.mean().item()

        # Margin (difference between top-2 logits)
        top2_logits = logits.topk(2, dim=1)[0]
        margin = (top2_logits[:, 0] - top2_logits[:, 1])
        mean_margin = margin.mean().item()

        # Loss
        loss = F.cross_entropy(logits, y).item()

    return {
        'accuracy': accuracy,
        'confidence': mean_confidence,
        'entropy': mean_entropy,
        'margin': mean_margin,
        'loss': loss,
    }


def verify_attacks(num_batches=5):
    """
    Verify that attacks work as intended.

    Expected behaviors:
    1. PGD: Increases loss, may reduce accuracy
    2. ConfSmooth: Reduces confidence, maintains 100% accuracy (backtracking)
    3. ClassAmbiguity: Reduces margin, maintains 100% accuracy (backtracking)
    """
    logger.info("="*60)
    logger.info("ATTACK VERIFICATION")
    logger.info("="*60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data
    logger.info("\nLoading CIFAR-10 test set...")
    _, test_loader = get_cifar10_loaders(batch_size=128, num_workers=0)

    # Initialize model (random weights for quick verification)
    logger.info("Initializing model...")
    model = get_resnet18_cifar10()
    model = model.to(device)
    model.eval()

    # Initialize attacks
    epsilon = 8/255
    alpha = 2/255

    pgd = PGDAttack(model, epsilon=epsilon, alpha=alpha, num_steps=10)
    confsmooth = ConfSmoothAttack(model, epsilon=epsilon, alpha=alpha, num_steps=20, num_classes=10)
    ambiguity = ClassPairAmbiguityAttack(model, epsilon=epsilon, alpha=alpha, num_steps=20, target_pair_mode="top2")

    logger.info(f"\nAttacks initialized:")
    logger.info(f"  - PGD: 10 steps")
    logger.info(f"  - ConfSmooth: 20 steps")
    logger.info(f"  - ClassAmbiguity: 20 steps")
    logger.info(f"  - Epsilon: {epsilon:.4f}")
    logger.info(f"  - Alpha: {alpha:.4f}")

    # Collect metrics across batches
    clean_metrics = []
    pgd_metrics = []
    confsmooth_metrics = []
    ambiguity_metrics = []

    logger.info(f"\nProcessing {num_batches} batches...")

    for batch_idx, (x, y) in enumerate(test_loader):
        if batch_idx >= num_batches:
            break

        logger.info(f"\nBatch {batch_idx + 1}/{num_batches}")
        x, y = x.to(device), y.to(device)

        # Clean metrics
        clean = compute_metrics(model, x, y, device)
        clean_metrics.append(clean)
        logger.info(f"  Clean: Acc={clean['accuracy']:.1f}%, Conf={clean['confidence']:.4f}, Ent={clean['entropy']:.4f}, Margin={clean['margin']:.4f}")

        # PGD attack
        x_pgd = pgd.generate(x, y)
        pgd_m = compute_metrics(model, x_pgd, y, device)
        pgd_metrics.append(pgd_m)
        logger.info(f"  PGD:   Acc={pgd_m['accuracy']:.1f}%, Conf={pgd_m['confidence']:.4f}, Ent={pgd_m['entropy']:.4f}, Loss={pgd_m['loss']:.4f}")

        # ConfSmooth attack
        x_conf = confsmooth.generate(x, y)
        conf_m = compute_metrics(model, x_conf, y, device)
        confsmooth_metrics.append(conf_m)
        logger.info(f"  ConfS: Acc={conf_m['accuracy']:.1f}%, Conf={conf_m['confidence']:.4f}, Ent={conf_m['entropy']:.4f}, Margin={conf_m['margin']:.4f}")

        # ClassAmbiguity attack
        x_amb = ambiguity.generate(x, y)
        amb_m = compute_metrics(model, x_amb, y, device)
        ambiguity_metrics.append(amb_m)
        logger.info(f"  Ambig: Acc={amb_m['accuracy']:.1f}%, Conf={amb_m['confidence']:.4f}, Ent={amb_m['entropy']:.4f}, Margin={amb_m['margin']:.4f}")

    # Average metrics
    def avg_metrics(metrics_list):
        return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in metrics_list[0].keys()}

    clean_avg = avg_metrics(clean_metrics)
    pgd_avg = avg_metrics(pgd_metrics)
    conf_avg = avg_metrics(confsmooth_metrics)
    amb_avg = avg_metrics(ambiguity_metrics)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*60)

    logger.info("\nAverage Metrics:")
    logger.info(f"{'Attack':<15} {'Accuracy':<12} {'Confidence':<12} {'Entropy':<12} {'Margin':<12} {'Loss':<12}")
    logger.info("-"*84)
    logger.info(f"{'Clean':<15} {clean_avg['accuracy']:>10.1f}% {clean_avg['confidence']:>11.4f} {clean_avg['entropy']:>11.4f} {clean_avg['margin']:>11.4f} {clean_avg['loss']:>11.4f}")
    logger.info(f"{'PGD':<15} {pgd_avg['accuracy']:>10.1f}% {pgd_avg['confidence']:>11.4f} {pgd_avg['entropy']:>11.4f} {pgd_avg['margin']:>11.4f} {pgd_avg['loss']:>11.4f}")
    logger.info(f"{'ConfSmooth':<15} {conf_avg['accuracy']:>10.1f}% {conf_avg['confidence']:>11.4f} {conf_avg['entropy']:>11.4f} {conf_avg['margin']:>11.4f} {conf_avg['loss']:>11.4f}")
    logger.info(f"{'ClassAmbiguity':<15} {amb_avg['accuracy']:>10.1f}% {amb_avg['confidence']:>11.4f} {amb_avg['entropy']:>11.4f} {amb_avg['margin']:>11.4f} {amb_avg['loss']:>11.4f}")

    # Verification checks
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION CHECKS")
    logger.info("="*60)

    checks_passed = 0
    total_checks = 0

    # Check 1: PGD increases loss
    total_checks += 1
    pgd_loss_increase = pgd_avg['loss'] > clean_avg['loss']
    status = "✅ PASS" if pgd_loss_increase else "❌ FAIL"
    logger.info(f"\n1. PGD increases loss: {status}")
    logger.info(f"   Clean loss: {clean_avg['loss']:.4f}, PGD loss: {pgd_avg['loss']:.4f}")
    if pgd_loss_increase:
        checks_passed += 1

    # Check 2: ConfSmooth reduces confidence
    total_checks += 1
    conf_reduces_confidence = conf_avg['confidence'] < clean_avg['confidence']
    status = "✅ PASS" if conf_reduces_confidence else "❌ FAIL"
    logger.info(f"\n2. ConfSmooth reduces confidence: {status}")
    logger.info(f"   Clean conf: {clean_avg['confidence']:.4f}, ConfSmooth conf: {conf_avg['confidence']:.4f}")
    logger.info(f"   Reduction: {(clean_avg['confidence'] - conf_avg['confidence']) / clean_avg['confidence'] * 100:.1f}%")
    if conf_reduces_confidence:
        checks_passed += 1

    # Check 3: ConfSmooth maintains 100% accuracy (backtracking works)
    total_checks += 1
    conf_maintains_accuracy = conf_avg['accuracy'] == 100.0
    status = "✅ PASS" if conf_maintains_accuracy else "❌ FAIL"
    logger.info(f"\n3. ConfSmooth maintains 100% accuracy (backtracking): {status}")
    logger.info(f"   ConfSmooth accuracy: {conf_avg['accuracy']:.1f}%")
    if conf_maintains_accuracy:
        checks_passed += 1

    # Check 4: ConfSmooth increases entropy (more uniform distribution)
    total_checks += 1
    conf_increases_entropy = conf_avg['entropy'] > clean_avg['entropy']
    status = "✅ PASS" if conf_increases_entropy else "❌ FAIL"
    logger.info(f"\n4. ConfSmooth increases entropy: {status}")
    logger.info(f"   Clean entropy: {clean_avg['entropy']:.4f}, ConfSmooth entropy: {conf_avg['entropy']:.4f}")
    logger.info(f"   Increase: {(conf_avg['entropy'] - clean_avg['entropy']) / clean_avg['entropy'] * 100:.1f}%")
    if conf_increases_entropy:
        checks_passed += 1

    # Check 5: ClassAmbiguity reduces confidence
    total_checks += 1
    amb_reduces_confidence = amb_avg['confidence'] < clean_avg['confidence']
    status = "✅ PASS" if amb_reduces_confidence else "❌ FAIL"
    logger.info(f"\n5. ClassAmbiguity reduces confidence: {status}")
    logger.info(f"   Clean conf: {clean_avg['confidence']:.4f}, Ambiguity conf: {amb_avg['confidence']:.4f}")
    logger.info(f"   Reduction: {(clean_avg['confidence'] - amb_avg['confidence']) / clean_avg['confidence'] * 100:.1f}%")
    if amb_reduces_confidence:
        checks_passed += 1

    # Check 6: ClassAmbiguity maintains 100% accuracy (backtracking works)
    total_checks += 1
    amb_maintains_accuracy = amb_avg['accuracy'] == 100.0
    status = "✅ PASS" if amb_maintains_accuracy else "❌ FAIL"
    logger.info(f"\n6. ClassAmbiguity maintains 100% accuracy (backtracking): {status}")
    logger.info(f"   Ambiguity accuracy: {amb_avg['accuracy']:.1f}%")
    if amb_maintains_accuracy:
        checks_passed += 1

    # Check 7: ClassAmbiguity reduces margin
    total_checks += 1
    amb_reduces_margin = amb_avg['margin'] < clean_avg['margin']
    status = "✅ PASS" if amb_reduces_margin else "❌ FAIL"
    logger.info(f"\n7. ClassAmbiguity reduces margin: {status}")
    logger.info(f"   Clean margin: {clean_avg['margin']:.4f}, Ambiguity margin: {amb_avg['margin']:.4f}")
    logger.info(f"   Reduction: {(clean_avg['margin'] - amb_avg['margin']) / clean_avg['margin'] * 100:.1f}%")
    if amb_reduces_margin:
        checks_passed += 1

    # Final summary
    logger.info("\n" + "="*60)
    logger.info(f"VERIFICATION RESULT: {checks_passed}/{total_checks} checks passed")
    logger.info("="*60)

    if checks_passed == total_checks:
        logger.info("\n✅ ALL CHECKS PASSED - Attacks working as intended!")
        return True
    else:
        logger.info(f"\n❌ {total_checks - checks_passed} checks failed - Review implementation")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify adversarial attacks")
    parser.add_argument("--num-batches", type=int, default=5, help="Number of batches to test")
    args = parser.parse_args()

    success = verify_attacks(num_batches=args.num_batches)
    sys.exit(0 if success else 1)
