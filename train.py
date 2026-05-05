# train.py
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import VesselDataset
from model import VesselNet
from utils import compute_loss

from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt

# =========================
# CHOOSE FIXED MATRIX SIZE HERE
# =========================
TARGET_HEIGHT = 300
TARGET_WIDTH = 298

# Other training settings
USE_PRECOMPUTED = True

TRAIN_DIR = "./data/train_data"
TEST_DIR = "./data/test_data"

TRAIN_DEMON_DIR = "./data/train_demon_mats"
TEST_DEMON_DIR = "./data/test_demon_mats"

# BATCH_SIZE = 16
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
# LEARNING_RATE = 3e-4
# NUM_EPOCHS = 10
# NUM_EPOCHS = 30
NUM_EPOCHS = 20
NUM_WORKERS = 4
# LAMBDA_COL = 1.0
# LAMBDA_COL = 0.5
LAMBDA_COL = 0.1
# LAMBDA_COL = 2.0


def save_run_config(run_dir, config_dict):
    config_path = run_dir / "run_config.txt"

    with open(config_path, "w") as f:
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")

def save_training_plots(history, run_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(run_dir / "loss_curve.png")
    plt.close()

    # Presence accuracy
    plt.figure()
    plt.plot(epochs, history["presence_acc"], label="presence_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Presence Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(run_dir / "presence_accuracy.png")
    plt.close()

    # Column accuracy
    plt.figure()
    plt.plot(epochs, history["column_exact"], label="column_exact")
    plt.plot(epochs, history["column_tol2"], label="column_tol2")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Column Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(run_dir / "column_accuracy.png")
    plt.close()

def create_run_dir(base_dir="./models"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = Path(base_dir) / f"model_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)

    for i, batch in enumerate(pbar):
        try:
            x = batch["x"].to(device)
            y_presence = batch["presence"].to(device)
            y_column = batch["column"].to(device)
        except Exception as e:
            print(f"\n[ERROR] Failed while loading batch {i+1}/{len(loader)}: {e}")
            raise

        optimizer.zero_grad()

        presence_logit, column_logits = model(x)

        loss, _, _ = compute_loss(
            presence_logit,
            column_logits,
            y_presence,
            y_column,
            lambda_col=LAMBDA_COL,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        pbar.set_postfix(loss=loss.item())

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()

    total_loss = 0.0
    correct_presence = 0
    total_presence = 0

    col_exact = 0
    col_tol1 = 0
    col_tol2 = 0
    col_tol5 = 0
    col_abs_error = 0.0
    col_count = 0

    for batch in loader:
        x = batch["x"].to(device)
        y_presence = batch["presence"].to(device)
        y_column = batch["column"].to(device)

        presence_logit, column_logits = model(x)

        loss, _, _ = compute_loss(
            presence_logit,
            column_logits,
            y_presence,
            y_column,
            lambda_col=LAMBDA_COL
        )
        total_loss += loss.item() * x.size(0)

        pred_presence = (torch.sigmoid(presence_logit) > 0.5).float()
        correct_presence += (pred_presence == y_presence).sum().item()
        total_presence += y_presence.numel()

        positive_mask = (y_presence == 1)
        if positive_mask.any():
            pred_column = column_logits[positive_mask].argmax(dim=1)
            true_column = y_column[positive_mask]

            abs_error = (pred_column - true_column).abs()

            col_exact += (abs_error == 0).sum().item()
            col_tol1 += (abs_error <= 1).sum().item()
            col_tol2 += (abs_error <= 2).sum().item()
            col_tol5 += (abs_error <= 5).sum().item()
            col_abs_error += abs_error.sum().item()
            col_count += true_column.numel()

    # print(f"Validation samples counted: {total_presence}")
    # print(f"Validation positive column samples counted: {col_count}")

    return {
    "loss": total_loss / len(loader.dataset),
    "presence_acc": correct_presence / total_presence if total_presence else 0.0,
    "column_exact": col_exact / col_count if col_count else 0.0,
    "column_tol1": col_tol1 / col_count if col_count else 0.0,
    "column_tol2": col_tol2 / col_count if col_count else 0.0,
    "column_tol5": col_tol5 / col_count if col_count else 0.0,
    "column_mae": col_abs_error / col_count if col_count else 0.0,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {torch.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config = {
        "NUM_EPOCHS": NUM_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "NUM_WORKERS": NUM_WORKERS,
        "TARGET_HEIGHT": TARGET_HEIGHT,
        "TARGET_WIDTH": TARGET_WIDTH,
        "TRAIN_DIR": TRAIN_DIR,
        "TEST_DIR": TEST_DIR,
        "USE_PRECOMPUTED": USE_PRECOMPUTED,
        "TRAIN_DEMON_DIR": TRAIN_DEMON_DIR,
        "TEST_DEMON_DIR": TEST_DEMON_DIR,
        "LAMBDA_COL": LAMBDA_COL,
    }
    config["COMBINED_SCORE_FORMULA"] = "presence_acc + 0.5 * column_tol2"
    config.update({
        "BLOCK_LEN": 1.0,
        "FPASS": 100,
        "FSTOP": 1500,
        "FPASS_DEMON": 300,
    })

    config.update({
        "DEVICE": str(device),
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "TORCH_VERSION": torch.__version__,
    })

    train_dataset = VesselDataset(
        data_dir=TRAIN_DIR,
        demon_dir=TRAIN_DEMON_DIR,
        use_precomputed=USE_PRECOMPUTED,
        target_height=TARGET_HEIGHT,
        target_width=TARGET_WIDTH,
        normalize=True,
    )

    val_dataset = VesselDataset(
        data_dir=TEST_DIR,
        demon_dir=TEST_DEMON_DIR,
        use_precomputed=USE_PRECOMPUTED,
        target_height=TARGET_HEIGHT,
        target_width=TARGET_WIDTH,
        normalize=True,
    )

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(val_dataset)}")
    print(f"Fixed matrix size: ({TARGET_HEIGHT}, {TARGET_WIDTH})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = VesselNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_acc = 0.0

    run_dir = create_run_dir("./models")
    print(f"Saving this run to: {run_dir}")
    save_run_config(run_dir, config)

    history = {
        "train_loss": [],
        "val_loss": [],
        "presence_acc": [],
        "column_exact": [],
        "column_tol1": [],
        "column_tol2": [],
        "column_tol5": [],
        "column_mae": [],
    }

    best_val_loss = float("inf")
    best_presence_acc = 0.0
    best_column_tol2 = 0.0
    best_combined_score = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{NUM_EPOCHS} =====", flush=True)

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)

        combined_score = (val_metrics["presence_acc"]+ 0.5 * val_metrics["column_tol2"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["presence_acc"].append(val_metrics["presence_acc"])
        history["column_exact"].append(val_metrics["column_exact"])
        history["column_tol2"].append(val_metrics["column_tol2"])
        history["column_tol1"].append(val_metrics["column_tol1"])
        history["column_tol5"].append(val_metrics["column_tol5"])
        history["column_mae"].append(val_metrics["column_mae"])

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"presence_acc={val_metrics['presence_acc']:.4f} | "
            f"col_exact={val_metrics['column_exact']:.4f} | "
            f"col_tol1={val_metrics['column_tol1']:.4f} | "
            f"col_tol2={val_metrics['column_tol2']:.4f} | "
            f"col_tol5={val_metrics['column_tol5']:.4f} | "
            f"col_mae={val_metrics['column_mae']:.2f}",
            flush=True,
        )

        # save latest model
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "target_height": TARGET_HEIGHT,
                "target_width": TARGET_WIDTH,
            },
            run_dir / "last_model.pth",
        )

        # save best model by validation loss
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                    "target_height": TARGET_HEIGHT,
                    "target_width": TARGET_WIDTH,
                    "best_val_loss": best_val_loss,
                },
                run_dir / "best_val_loss_model.pth",
            )

            print("Saved best model!", flush=True)

        # Save best presence model
        if val_metrics["presence_acc"] > best_presence_acc:
            best_presence_acc = val_metrics["presence_acc"]

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metric": best_presence_acc,
                    "type": "best_presence_acc",
                },
                run_dir / "best_presence_acc.pth",
            )

            print("Saved best_presence_acc model!", flush=True)


        # Save best column_tol2 model
        if val_metrics["column_tol2"] > best_column_tol2:
            best_column_tol2 = val_metrics["column_tol2"]

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metric": best_column_tol2,
                    "type": "best_column_tol2",
                },
                run_dir / "best_column_tol2.pth",
            )

            print("Saved best_column_tol2 model!", flush=True)


        # Save best combined model
        if combined_score > best_combined_score:
            best_combined_score = combined_score

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metric": best_combined_score,
                    "type": "best_combined",
                },
                run_dir / "best_combined.pth",
            )

            print("Saved best_combined model!", flush=True)

        # save metrics JSON
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=4)

        # save plots every epoch
        save_training_plots(history, run_dir)


if __name__ == "__main__":
    main()