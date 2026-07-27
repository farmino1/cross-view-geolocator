import numpy as np
import torch

from src.training.losses import DistillationLoss, SymmetricInfoNCE


class TestSymmetricInfoNCE:
    def test_perfect_pairs_have_low_loss(self):
        """When satellite and phone embeddings are identical, loss should be low."""
        N, D = 32, 256
        emb = torch.randn(N, D)
        emb = torch.nn.functional.normalize(emb, dim=1)

        criterion = SymmetricInfoNCE(temperature_init=0.07)
        loss = criterion(emb, emb)
        assert loss.item() < 0.1, f"Loss too high for perfect pairs: {loss.item()}"

    def test_random_pairs_have_high_loss(self):
        """Random embeddings should have loss around ln(N)."""
        N, D = 32, 256
        criterion = SymmetricInfoNCE(temperature_init=0.07)

        losses = []
        for _ in range(5):
            emb1 = torch.nn.functional.normalize(torch.randn(N, D), dim=1)
            emb2 = torch.nn.functional.normalize(torch.randn(N, D), dim=1)
            loss = criterion(emb1, emb2)
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        expected = np.log(N)  # ln(32) ≈ 3.47
        # With temperature scaling, actual loss differs but should be > ln(N)/2
        assert avg_loss > 1.0, f"Loss too low for random pairs: {avg_loss}"

    def test_loss_decreases_with_learning(self):
        """Loss should decrease when we move correct pairs closer."""
        N, D = 16, 128
        torch.manual_seed(42)

        sat = torch.randn(N, D, requires_grad=True)
        phone = torch.randn(N, D, requires_grad=True)

        criterion = SymmetricInfoNCE(temperature_init=0.07)
        optimizer = torch.optim.Adam([sat, phone], lr=0.01)

        initial_loss = criterion(
            torch.nn.functional.normalize(sat, dim=1),
            torch.nn.functional.normalize(phone, dim=1),
        ).item()

        for _ in range(50):
            optimizer.zero_grad()
            sat_norm = torch.nn.functional.normalize(sat, dim=1)
            phone_norm = torch.nn.functional.normalize(phone, dim=1)
            loss = criterion(sat_norm, phone_norm)
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss, f"Loss didn't decrease: {initial_loss} -> {final_loss}"

    def test_temperature_is_learnable(self):
        """Temperature parameter should be in optimizer."""
        criterion = SymmetricInfoNCE(temperature_init=0.07)
        params = list(criterion.parameters())
        assert len(params) == 1  # log_temperature

        optimizer = torch.optim.Adam(params, lr=0.01)
        initial_temp = criterion.temperature.item()

        for _ in range(10):
            optimizer.zero_grad()
            emb = torch.nn.functional.normalize(torch.randn(8, 64), dim=1)
            loss = criterion(emb, emb)
            loss.backward()
            optimizer.step()

        final_temp = criterion.temperature.item()
        assert final_temp != initial_temp, "Temperature didn't change"


class TestDistillationLoss:
    def test_identical_embeddings_give_zero_loss(self):
        N, D = 32, 256
        emb = torch.randn(N, D)
        criterion = DistillationLoss()
        loss = criterion(emb, emb)
        assert loss.item() < 1e-6

    def test_different_embeddings_give_positive_loss(self):
        N, D = 32, 256
        emb1 = torch.randn(N, D)
        emb2 = torch.randn(N, D)
        criterion = DistillationLoss()
        loss = criterion(emb1, emb2)
        assert loss.item() > 0

    def test_loss_decreases_with_learning(self):
        N, D = 16, 128
        teacher_emb = torch.randn(N, D)
        student = torch.nn.Linear(64, 128)  # Simplified
        criterion = DistillationLoss()
        optimizer = torch.optim.Adam(student.parameters(), lr=0.01)

        # This test is simplified - in real distillation, teacher and student
        # have different architectures. Here we just verify the loss works.
        x = torch.randn(N, 64)
        initial_loss = criterion(student(x), teacher_emb).item()

        for _ in range(50):
            optimizer.zero_grad()
            loss = criterion(student(x), teacher_emb)
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss
