import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import CVCitiesDataset
from .losses import DistillationLoss
from .models import create_mobilenetv3_encoder, create_resnet50_encoder


def train_distillation_epoch(
    teacher: nn.Module,
    student: nn.Module,
    dataloader: DataLoader,
    criterion: DistillationLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Train for one epoch. Returns average loss."""
    teacher.eval()
    student.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        images = batch["phone"].to(device)  # Student must encode phone views

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                with torch.no_grad():
                    teacher_emb = teacher(images)
                student_emb = student(images)
                loss = criterion(student_emb, teacher_emb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                teacher_emb = teacher(images)
            student_emb = student(images)
            loss = criterion(student_emb, teacher_emb)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def distill(
    teacher_checkpoint: str,
    data_dir: str,
    cities: list[str],
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-4,
    embed_dim: int = 256,
    resume_from: str = None,
    device: str = "auto",
):
    """
    Distill ResNet-50 teacher to MobileNetV3 student.

    The teacher's phone encoder embeddings are used as targets.
    The student learns to produce the same embeddings on phone images.
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load trained teacher
    checkpoint = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
    teacher = create_resnet50_encoder(embed_dim=embed_dim, pretrained=False).to(device)
    teacher.load_state_dict(checkpoint["phone_encoder"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Create student
    student = create_mobilenetv3_encoder(embed_dim=embed_dim, pretrained=True).to(device)

    # Loss and optimizer
    criterion = DistillationLoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.1)

    # Cosine schedule
    def lr_lambda(step):
        progress = step / max(1, epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    start_epoch = 0
    best_loss = float("inf")

    # Resume from checkpoint
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["student"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "best_loss" in ckpt:
            best_loss = ckpt["best_loss"]
        start_epoch = ckpt.get("epoch", 0)
        print(f"  Resumed from epoch {start_epoch}, best loss: {best_loss:.6f}")

    # Dataset (only satellite images needed)
    dataset = CVCitiesDataset(
        data_dir,
        cities,
        phone_augmentation=True,
        satellite_augmentation=True,
        val_split=0.1,
        is_val=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Distillation: {len(dataset)} samples, Teacher: ResNet-50 → Student: MobileNetV3")
    print(f"Device: {device}")

    for epoch in range(start_epoch, epochs):
        start = time.time()
        loss = train_distillation_epoch(
            teacher, student, loader, criterion, optimizer, device, scaler
        )
        scheduler.step()
        elapsed = time.time() - start

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.6f} | Time: {elapsed:.1f}s")

        # Save resume checkpoint every epoch
        torch.save(
            {
                "student": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss,
                "epoch": epoch + 1,
            },
            output_path / "latest_student.pt",
        )

        if loss < best_loss:
            best_loss = loss
            torch.save(
                {
                    "student": student.state_dict(),
                    "loss": loss,
                    "epoch": epoch + 1,
                },
                output_path / "best_student.pt",
            )

    # Save final
    torch.save(
        {
            "student": student.state_dict(),
            "loss": loss,
            "epoch": epochs,
        },
        output_path / "final_student.pt",
    )

    print(f"\nDistillation complete. Best loss: {best_loss:.6f}")
    print(f"Student model saved to {output_path}")

    return student


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill ResNet-50 to MobileNetV3")
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cities", type=str, nargs="+", default=["seattle", "london", "tokyo", "sydney"])
    parser.add_argument("--output_dir", type=str, default="./checkpoints/student")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    distill(
        teacher_checkpoint=args.teacher_checkpoint,
        data_dir=args.data_dir,
        cities=args.cities,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume_from=args.resume_from,
        device=args.device,
    )
