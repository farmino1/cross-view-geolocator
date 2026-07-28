import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .augmentation import get_phone_augmentation, get_satellite_augmentation
from .dataset import CVCitiesDataset, extract_perspective_crops
from .losses import SymmetricInfoNCE
from .models import create_resnet50_encoder


def compute_recall_at_k(
    sat_embeddings: torch.Tensor,
    phone_embeddings: torch.Tensor,
    k_values: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """
    Compute recall@K for cross-view matching.

    For each phone query, check if the correct satellite image
    appears in the top-K retrieved results.
    """
    # Similarity matrix: phone queries vs satellite gallery
    similarities = phone_embeddings @ sat_embeddings.T  # [N_query, N_gallery]
    num_gallery = sat_embeddings.shape[0]

    # For each query, rank gallery items by similarity
    _, sorted_indices = similarities.sort(dim=1, descending=True)

    # Labels: each query matches the gallery item at the same index
    labels = torch.arange(num_gallery, device=similarities.device)

    results = {}
    for k in k_values:
        if k > num_gallery:
            results[f"recall@{k}"] = 0.0
            continue
        top_k = sorted_indices[:, :k]
        # Check if correct match is in top-K
        correct = (top_k == labels.unsqueeze(1)).any(dim=1)
        results[f"recall@{k}"] = correct.float().mean().item()

    return results


def train_epoch(
    sat_encoder: nn.Module,
    phone_encoder: nn.Module,
    dataloader: DataLoader,
    criterion: SymmetricInfoNCE,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Train for one epoch. Returns average loss."""
    sat_encoder.train()
    phone_encoder.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        sat_images = batch["satellite"].to(device)
        phone_images = batch["phone"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                sat_emb = sat_encoder(sat_images)
                phone_emb = phone_encoder(phone_images)
                loss = criterion(sat_emb, phone_emb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(sat_encoder.parameters()) + list(phone_encoder.parameters()),
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            sat_emb = sat_encoder(sat_images)
            phone_emb = phone_encoder(phone_images)
            loss = criterion(sat_emb, phone_emb)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(sat_encoder.parameters()) + list(phone_encoder.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def validate(
    sat_encoder: nn.Module,
    phone_encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    k_values: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """Validate and compute recall@K metrics."""
    sat_encoder.eval()
    phone_encoder.eval()

    all_sat_emb = []
    all_phone_emb = []

    for batch in dataloader:
        sat_images = batch["satellite"].to(device)
        phone_images = batch["phone"].to(device)

        sat_emb = sat_encoder(sat_images)
        phone_emb = phone_encoder(phone_images)

        all_sat_emb.append(sat_emb)
        all_phone_emb.append(phone_emb)

    all_sat_emb = torch.cat(all_sat_emb, dim=0)
    all_phone_emb = torch.cat(all_phone_emb, dim=0)

    return compute_recall_at_k(all_sat_emb, all_phone_emb, k_values)


def train(
    data_dir: str,
    cities: list[str],
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 0.2,
    embed_dim: int = 256,
    warmup_epochs: int = 1,
    checkpoint_interval: int = 5,
    resume_from: str = None,
    device: str = "auto",
):
    """
    Train satellite and phone encoders with symmetric InfoNCE loss.

    Args:
        data_dir: path to CV-Cities data directory
        cities: list of city names to use for training
        output_dir: where to save checkpoints and logs
        epochs: number of training epochs (total, not additional)
        batch_size: batch size
        lr: learning rate
        weight_decay: weight decay for AdamW
        embed_dim: embedding dimension (256)
        warmup_epochs: number of warmup epochs
        checkpoint_interval: save checkpoint every N epochs
        resume_from: path to checkpoint to resume from
        device: 'auto', 'cuda', or 'cpu'
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create encoders
    sat_encoder = create_resnet50_encoder(embed_dim=embed_dim, pretrained=True).to(
        device
    )
    phone_encoder = create_resnet50_encoder(embed_dim=embed_dim, pretrained=True).to(
        device
    )

    # Loss and optimizer
    criterion = SymmetricInfoNCE(temperature_init=0.07).to(device)

    params = (
        list(sat_encoder.parameters())
        + list(phone_encoder.parameters())
        + list(criterion.parameters())
    )
    optimizer = torch.optim.AdamW(
        params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )

    # Cosine LR schedule with warmup
    def lr_lambda(step):
        if step < warmup_epochs:
            return (step + 1) / warmup_epochs
        progress = (step - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler for mixed precision
    scaler = (
        torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    )

    start_epoch = 0
    best_recall_1 = 0.0
    history = []

    # Resume from checkpoint
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        sat_encoder.load_state_dict(ckpt["sat_encoder"])
        phone_encoder.load_state_dict(ckpt["phone_encoder"])
        criterion.load_state_dict(ckpt["criterion"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "history" in ckpt:
            history = ckpt["history"]
            for h in history:
                if h.get("recall@1", 0) > best_recall_1:
                    best_recall_1 = h["recall@1"]
        start_epoch = ckpt.get("epoch", 0)
        print(f"  Resumed from epoch {start_epoch}, best R@1: {best_recall_1:.4f}")

    # Datasets
    train_dataset = CVCitiesDataset(
        data_dir,
        cities,
        phone_augmentation=True,
        satellite_augmentation=True,
        val_split=0.2,
        is_val=False,
    )
    val_dataset = CVCitiesDataset(
        data_dir,
        cities,
        phone_augmentation=False,
        satellite_augmentation=False,
        val_split=0.2,
        is_val=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples")
    print(f"Device: {device}, Batch size: {batch_size}, Epochs: {start_epoch}→{epochs}")
    print(f"Temperature init: {criterion.temperature.item():.4f}")

    for epoch in range(start_epoch, epochs):
        start = time.time()

        train_loss = train_epoch(
            sat_encoder, phone_encoder, train_loader, criterion, optimizer, device, scaler
        )
        scheduler.step()

        # Validate every 5 epochs or on last epoch
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            metrics = validate(sat_encoder, phone_encoder, val_loader, device)
            elapsed = time.time() - start

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"R@1: {metrics['recall@1']:.4f} | "
                f"R@5: {metrics['recall@5']:.4f} | "
                f"R@10: {metrics['recall@10']:.4f} | "
                f"Temp: {criterion.temperature.item():.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": train_loss,
                    **metrics,
                    "temperature": criterion.temperature.item(),
                }
            )

            # Save best model
            if metrics["recall@1"] > best_recall_1:
                best_recall_1 = metrics["recall@1"]
                torch.save(
                    {
                        "sat_encoder": sat_encoder.state_dict(),
                        "phone_encoder": phone_encoder.state_dict(),
                        "criterion": criterion.state_dict(),
                        "metrics": metrics,
                        "epoch": epoch + 1,
                    },
                    output_path / "best_model.pt",
                )
                print(f"  → New best R@1: {best_recall_1:.4f}")
        else:
            elapsed = time.time() - start
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Temp: {criterion.temperature.item():.4f} | "
                f"Time: {elapsed:.1f}s"
            )

        # Save resume checkpoint every epoch
        torch.save(
            {
                "sat_encoder": sat_encoder.state_dict(),
                "phone_encoder": phone_encoder.state_dict(),
                "criterion": criterion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "history": history,
            },
            output_path / "latest_checkpoint.pt",
        )

        # Periodic full checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(
                {
                    "sat_encoder": sat_encoder.state_dict(),
                    "phone_encoder": phone_encoder.state_dict(),
                    "criterion": criterion.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "history": history,
                },
                output_path / f"checkpoint_epoch{epoch+1}.pt",
            )

    # Save final model
    torch.save(
        {
            "sat_encoder": sat_encoder.state_dict(),
            "phone_encoder": phone_encoder.state_dict(),
            "criterion": criterion.state_dict(),
            "metrics": history[-1] if history else {},
            "epoch": epochs,
        },
        output_path / "final_model.pt",
    )

    # Save training history
    with open(output_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best R@1: {best_recall_1:.4f}")
    print(f"Models saved to {output_path}")

    return sat_encoder, phone_encoder, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train cross-view encoders")
    parser.add_argument("--data_dir", type=str, required=True, help="CV-Cities data directory")
    parser.add_argument(
        "--cities",
        type=str,
        nargs="+",
        default=["seattle", "london", "tokyo", "sydney"],
    )
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.2)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        cities=args.cities,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        embed_dim=args.embed_dim,
        device=args.device,
    )
