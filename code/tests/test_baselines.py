import torch

from models.baseline_models import build_baseline


def test_baseline_shapes():
    x = torch.randn(2, 40, 4)
    for name in [
        "lstm",
        "seq2seq_lstm",
        "attention_lstm",
        "transformer",
    ]:
        model = build_baseline(
            name,
            input_dimension=4,
            prediction_horizon=20,
            **(
                {
                    "hidden_dimension": 64,
                    "number_of_layers": 1,
                    "dropout": 0.0,
                }
                if name != "transformer"
                else {
                    "latent_dimension": 64,
                    "number_of_layers": 1,
                    "number_of_attention_heads": 8,
                    "feedforward_dimension": 128,
                    "dropout": 0.0,
                }
            ),
        )
        trajectory, cri = model(x)
        assert trajectory.shape == (2, 20, 2)
        assert cri.shape == (2, 20, 1)
