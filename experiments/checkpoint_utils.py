"""Utilities for managing and inspecting checkpoints."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch


def list_checkpoints(directory: str, pattern: str = "*.pt") -> List[Path]:
    """
    List all checkpoints in a directory.

    Args:
        directory: Directory to search
        pattern: Glob pattern for checkpoint files

    Returns:
        List of checkpoint paths sorted by modification time
    """
    directory = Path(directory)

    if not directory.exists():
        print(f"Directory not found: {directory}")
        return []

    checkpoints = sorted(
        directory.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,  # Most recent first
    )

    return checkpoints


def inspect_checkpoint(checkpoint_path: str) -> Dict:
    """
    Inspect a checkpoint and return metadata.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    metadata = {
        "path": str(checkpoint_path),
        "size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
        "epoch": checkpoint.get("epoch", "N/A"),
        "has_history": "history" in checkpoint,
        "has_config": "config" in checkpoint,
        "has_optimizer": "optimizer_state_dict" in checkpoint,
    }

    # Add history summary if available
    if "history" in checkpoint:
        history = checkpoint["history"]
        if history["val_acc"]:
            metadata["final_val_acc"] = history["val_acc"][-1]
            metadata["best_val_acc"] = max(history["val_acc"])
            metadata["best_epoch"] = history["val_acc"].index(metadata["best_val_acc"]) + 1

    # Add best_val_acc if saved in checkpoint
    if "best_val_acc" in checkpoint:
        metadata["saved_best_val_acc"] = checkpoint["best_val_acc"]

    return metadata


def print_checkpoint_table(checkpoints: List[Path], verbose: bool = False):
    """
    Print a formatted table of checkpoints.

    Args:
        checkpoints: List of checkpoint paths
        verbose: Whether to show detailed information
    """
    if not checkpoints:
        print("No checkpoints found.")
        return

    print("\n" + "="*120)
    print(f"Found {len(checkpoints)} checkpoint(s)")
    print("="*120)

    # Header
    if verbose:
        print(f"{'Filename':<40} {'Epoch':<8} {'Size (MB)':<10} {'Val Acc':<10} {'Best Acc':<10} {'History':<8} {'Config':<8}")
        print("-"*120)
    else:
        print(f"{'Filename':<40} {'Epoch':<8} {'Size (MB)':<10} {'Val Acc':<10} {'Best Acc':<10}")
        print("-"*120)

    # Checkpoint rows
    for ckpt_path in checkpoints:
        try:
            meta = inspect_checkpoint(ckpt_path)

            filename = ckpt_path.name
            epoch = str(meta["epoch"])
            size = f"{meta['size_mb']:.1f}"
            val_acc = f"{meta.get('final_val_acc', 'N/A'):.2f}%" if isinstance(meta.get('final_val_acc'), (int, float)) else "N/A"
            best_acc = f"{meta.get('best_val_acc', 'N/A'):.2f}%" if isinstance(meta.get('best_val_acc'), (int, float)) else "N/A"

            if verbose:
                has_hist = "✓" if meta["has_history"] else "✗"
                has_conf = "✓" if meta["has_config"] else "✗"
                print(f"{filename:<40} {epoch:<8} {size:<10} {val_acc:<10} {best_acc:<10} {has_hist:<8} {has_conf:<8}")
            else:
                print(f"{filename:<40} {epoch:<8} {size:<10} {val_acc:<10} {best_acc:<10}")

        except Exception as e:
            print(f"{ckpt_path.name:<40} ERROR: {e}")

    print("="*120 + "\n")


def find_best_checkpoint(directory: str, pattern: str = "*.pt") -> Optional[Path]:
    """
    Find the checkpoint with the best validation accuracy.

    Args:
        directory: Directory to search
        pattern: Glob pattern for checkpoint files

    Returns:
        Path to best checkpoint, or None if not found
    """
    checkpoints = list_checkpoints(directory, pattern)

    if not checkpoints:
        return None

    best_ckpt = None
    best_acc = -1.0

    for ckpt_path in checkpoints:
        try:
            meta = inspect_checkpoint(ckpt_path)
            acc = meta.get("best_val_acc") or meta.get("final_val_acc", -1.0)

            if isinstance(acc, (int, float)) and acc > best_acc:
                best_acc = acc
                best_ckpt = ckpt_path

        except Exception:
            continue

    return best_ckpt


def compare_checkpoints(checkpoint_paths: List[str]):
    """
    Compare multiple checkpoints side-by-side.

    Args:
        checkpoint_paths: List of checkpoint paths to compare
    """
    print("\n" + "="*100)
    print("CHECKPOINT COMPARISON")
    print("="*100)

    metadata_list = []
    for path in checkpoint_paths:
        try:
            meta = inspect_checkpoint(path)
            metadata_list.append(meta)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

    if not metadata_list:
        print("No valid checkpoints to compare.")
        return

    # Print comparison table
    print(f"\n{'Metric':<30}", end="")
    for i in range(len(metadata_list)):
        print(f"Checkpoint {i+1:<15}", end="")
    print("\n" + "-"*100)

    # Filename
    print(f"{'Filename':<30}", end="")
    for meta in metadata_list:
        filename = Path(meta["path"]).name[:15]
        print(f"{filename:<20}", end="")
    print()

    # Epoch
    print(f"{'Epoch':<30}", end="")
    for meta in metadata_list:
        print(f"{str(meta['epoch']):<20}", end="")
    print()

    # Size
    print(f"{'Size (MB)':<30}", end="")
    for meta in metadata_list:
        print(f"{meta['size_mb']:.1f}{'':<17}", end="")
    print()

    # Final Val Acc
    if all("final_val_acc" in m for m in metadata_list):
        print(f"{'Final Val Accuracy (%)':<30}", end="")
        for meta in metadata_list:
            acc = meta.get("final_val_acc", "N/A")
            if isinstance(acc, (int, float)):
                print(f"{acc:.2f}{'':<16}", end="")
            else:
                print(f"{'N/A':<20}", end="")
        print()

    # Best Val Acc
    if all("best_val_acc" in m for m in metadata_list):
        print(f"{'Best Val Accuracy (%)':<30}", end="")
        for meta in metadata_list:
            acc = meta.get("best_val_acc", "N/A")
            if isinstance(acc, (int, float)):
                print(f"{acc:.2f}{'':<16}", end="")
            else:
                print(f"{'N/A':<20}", end="")
        print()

        print(f"{'Best Epoch':<30}", end="")
        for meta in metadata_list:
            epoch = meta.get("best_epoch", "N/A")
            print(f"{str(epoch):<20}", end="")
        print()

    print("="*100 + "\n")


def cleanup_old_checkpoints(
    directory: str,
    keep_n: int = 5,
    keep_best: bool = True,
    keep_final: bool = True,
    dry_run: bool = True,
):
    """
    Clean up old checkpoints, keeping only the most recent.

    Args:
        directory: Directory containing checkpoints
        keep_n: Number of recent periodic checkpoints to keep
        keep_best: Whether to keep best_model.pt
        keep_final: Whether to keep final_model.pt
        dry_run: If True, only print what would be deleted
    """
    directory = Path(directory)
    all_checkpoints = list_checkpoints(directory)

    # Separate checkpoints by type
    periodic = []
    best = []
    final = []

    for ckpt in all_checkpoints:
        if "best_model" in ckpt.name:
            best.append(ckpt)
        elif "final_model" in ckpt.name:
            final.append(ckpt)
        elif "epoch" in ckpt.name:
            periodic.append(ckpt)

    # Sort periodic by epoch number (most recent first)
    periodic.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Determine what to delete
    to_delete = []

    # Delete old periodic checkpoints
    if len(periodic) > keep_n:
        to_delete.extend(periodic[keep_n:])

    # Delete old best models (keep only most recent)
    if not keep_best and best:
        to_delete.extend(best)
    elif len(best) > 1:
        to_delete.extend(best[1:])

    # Delete old final models (keep only most recent)
    if not keep_final and final:
        to_delete.extend(final)
    elif len(final) > 1:
        to_delete.extend(final[1:])

    # Print summary
    print(f"\n{'='*60}")
    print(f"Cleanup Summary ({'DRY RUN' if dry_run else 'ACTUAL'})")
    print(f"{'='*60}")
    print(f"Directory: {directory}")
    print(f"Total checkpoints: {len(all_checkpoints)}")
    print(f"  - Periodic: {len(periodic)}")
    print(f"  - Best: {len(best)}")
    print(f"  - Final: {len(final)}")
    print(f"\nKeeping:")
    print(f"  - {keep_n} most recent periodic checkpoints")
    print(f"  - Best model: {keep_best}")
    print(f"  - Final model: {keep_final}")
    print(f"\nWould delete: {len(to_delete)} checkpoint(s)")

    if to_delete:
        total_size = sum(p.stat().st_size for p in to_delete) / (1024 * 1024)
        print(f"Space to reclaim: {total_size:.1f} MB")
        print(f"\nFiles to delete:")
        for ckpt in to_delete:
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            print(f"  - {ckpt.name} ({size_mb:.1f} MB)")

    print(f"{'='*60}\n")

    # Perform deletion if not dry run
    if not dry_run and to_delete:
        confirm = input("Proceed with deletion? (yes/no): ")
        if confirm.lower() == "yes":
            for ckpt in to_delete:
                ckpt.unlink()
                print(f"Deleted: {ckpt.name}")
            print(f"\nDeleted {len(to_delete)} checkpoint(s)")
        else:
            print("Deletion cancelled")


def main():
    parser = argparse.ArgumentParser(description="Checkpoint management utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    list_parser = subparsers.add_parser("list", help="List checkpoints in directory")
    list_parser.add_argument("directory", type=str, help="Directory containing checkpoints")
    list_parser.add_argument("--pattern", type=str, default="*.pt", help="Glob pattern (default: *.pt)")
    list_parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed information")

    # Best command
    best_parser = subparsers.add_parser("best", help="Find best checkpoint")
    best_parser.add_argument("directory", type=str, help="Directory containing checkpoints")
    best_parser.add_argument("--pattern", type=str, default="*.pt", help="Glob pattern (default: *.pt)")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple checkpoints")
    compare_parser.add_argument("checkpoints", type=str, nargs="+", help="Checkpoint paths to compare")

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old checkpoints")
    cleanup_parser.add_argument("directory", type=str, help="Directory containing checkpoints")
    cleanup_parser.add_argument("--keep", type=int, default=5, help="Number of recent checkpoints to keep (default: 5)")
    cleanup_parser.add_argument("--no-keep-best", action="store_true", help="Don't keep best_model.pt")
    cleanup_parser.add_argument("--no-keep-final", action="store_true", help="Don't keep final_model.pt")
    cleanup_parser.add_argument("--execute", action="store_true", help="Actually delete (default is dry run)")

    args = parser.parse_args()

    if args.command == "list":
        checkpoints = list_checkpoints(args.directory, args.pattern)
        print_checkpoint_table(checkpoints, verbose=args.verbose)

    elif args.command == "best":
        best_ckpt = find_best_checkpoint(args.directory, args.pattern)
        if best_ckpt:
            print(f"\nBest checkpoint: {best_ckpt}")
            print("\nMetadata:")
            meta = inspect_checkpoint(best_ckpt)
            for key, value in meta.items():
                print(f"  {key}: {value}")
        else:
            print("No checkpoints found.")

    elif args.command == "compare":
        compare_checkpoints(args.checkpoints)

    elif args.command == "cleanup":
        cleanup_old_checkpoints(
            args.directory,
            keep_n=args.keep,
            keep_best=not args.no_keep_best,
            keep_final=not args.no_keep_final,
            dry_run=not args.execute,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
