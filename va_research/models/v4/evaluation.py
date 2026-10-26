"""Offline V4 evaluation with optional diagnostic plots."""

import argparse
from pathlib import Path
import numpy as np
import torch

from .model import V4Policy
from .dataset import CanonicalV4, dump
from .training import predict, metrics, passes


def load(checkpoint, device="cpu"):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = V4Policy().to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved


def evaluate(checkpoint, output):
    torch.set_num_threads(2)
    model, saved = load(checkpoint)
    dataset = CanonicalV4(saved["manifest"])
    data = dataset.data
    if dataset.provenance != saved["provenance"]:
        raise ValueError("Checkpoint dataset changed")
    prediction = predict(model, data)
    target = {key: value.numpy() for key, value in data.items()}
    predicted = {key: value.numpy() for key, value in prediction.items()}
    result = metrics(prediction, data, dataset.moving)
    body_error = np.abs(predicted["body_tokens"] - target["body_tokens"])
    hand_error = np.abs(predicted["hand_ref"] - target["hand_ref"])
    result.update(body_mae_by_slot=body_error.mean((0, 2)).tolist(), hand_mae_by_slot=hand_error.mean((0, 2)).tolist(),
                  offline_gate=passes(result, saved["config"]["offline_gate"]), epoch=saved["epoch"],
                  single_source=True, plots_generated=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(result["body_mae_by_slot"]); axes[0].set_title("Body token MAE")
        axes[1].plot(result["hand_mae_by_slot"]); axes[1].set_title("Hand reference MAE")
        fig.tight_layout(); fig.savefig(output / "horizon_errors.png"); plt.close(fig)
        result["plots_generated"] = True
    except ModuleNotFoundError:
        result["plot_warning"] = "matplotlib unavailable; metrics remain valid"
    dump(output / "offline_metrics.json", result)
    np.savez_compressed(output / "offline_predictions.npz", **predicted, sample_indices=dataset.keep,
                        target_body=target["body_tokens"], target_hand=target["hand_ref"],
                        target_phase=target["phase"], target_progress=target["progress"])
    print(result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
