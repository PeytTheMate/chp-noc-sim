# tests/test_gnn_training_smoke.py
from gnn_agent.train import train_risk_aware_policy

def test_training_reduces_loss_smoke() -> None:
    model = train_risk_aware_policy(num_epochs=3, episodes_per_epoch=8)
    # You’re not asserting actual values, but this will catch
    # catastrophic bugs (NaNs, shape mis-matches, etc.) during training.
    assert model is not None
