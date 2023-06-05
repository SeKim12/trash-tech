import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.pipeline import generate_split
from models import get_model, train_model, test_model
import seaborn as sns
import matplotlib.pyplot as plt
import json
import datetime
import os

START_DATE = datetime.datetime.now().strftime("%Y-%m-%d")
TEST_RESULTS_DIR = "test_results"
TRAIN_RESULTS_DIR = "train_results"
CHECKPOINT_DIR = "checkpoint"
DATA_DIR = "data/dataset-resized"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate")
    args = parser.parse_args()

    # TODO: Add MobileNet
    mdls = [args.model] if args.model != "all" else ["resnet", "vgg16", "vit"]

    # perform a grid search over all combinations of hyperparameters
    learning_rates = [0.1, 0.01, 0.001] if not args.debug else [0.1]
    dropout_rates = [0.1, 0.2, 0.3] if not args.debug else [0.1]
    initializers = ["xavier", "he", "normal"] if not args.debug else ["xavier"]

    args.epochs = args.epochs if not args.debug else 3

    criterion = nn.CrossEntropyLoss()

    # TODO: add distribution shift data
    dataset = generate_split(args.data_dir)
    train_loader = DataLoader(dataset["train"], batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset["val"], batch_size=16, shuffle=True)

    # reduce dataset size for debugging purposes
    if args.debug:
        train_loader = val_loader

    test_loader = DataLoader(dataset["test"], batch_size=16, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # save all results from all hyperparameters
    results = {}

    for mdl in mdls:
        print(f"Starting Training for {mdl}...")

        # save best model (across all hyperparameters)
        glob_best_val_acc = -1

        for lr in learning_rates:
            for dropout_rate in dropout_rates:
                for initializer in initializers:
                    model = get_model(mdl, dropout_rate, initializer).to(device)
                    print(
                        f"=====> Testing LR: {lr}, Dropout Rate: {dropout_rate}, Initializer: {initializer}"
                    )
                    config = {
                        "checkpoint_dir": CHECKPOINT_DIR,
                        "model_name": mdl,
                        "learning_rate": lr,
                        "dropout_rate": dropout_rate,
                        "initializer": initializer,
                    }
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    (
                        train_losses,
                        val_losses,
                        val_metrics,
                        glob_best_val_acc,
                    ) = train_model(
                        model,
                        config,
                        criterion,
                        optimizer,
                        train_loader,
                        val_loader,
                        args.epochs,
                        device,
                        glob_best_val_acc,
                    )

                    # overall
                    # test_metrics, test_loss, conf_matrix = test_model(model, criterion, test_loader)

                    results[mdl] = results.get(mdl, {})
                    results[mdl][
                        f"lr={lr}, dropout_rate={dropout_rate}, initializer={initializer}"
                    ] = {
                        "learning_rate": lr,
                        "dropout_rate": dropout_rate,
                        "initializer": initializer,
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "val_metrics": val_metrics,
                    }

                    # record best so far
                    # this is for convenience, the results are a duplicate of the above
                    results[mdl]["best"] = results[mdl].get(
                        "best", {"val_metrics": {"accuracy": -1}}
                    )
                    if (
                        val_metrics["accuracy"]
                        > results[mdl]["best"]["val_metrics"]["accuracy"]
                    ):
                        results[mdl]["best"] = {
                            "learning_rate": lr,
                            "dropout_rate": dropout_rate,
                            "initializer": initializer,
                            "train_losses": train_losses,
                            "val_losses": val_losses,
                            "val_metrics": val_metrics,
                        }

    print("Training complete.")

    # save train resuts
    for mdl in results:
        print(f"Best Results for {mdl}")
        print(results[mdl]["best"])
        print("")
        dir_path = os.path.join(mdl, START_DATE, TRAIN_RESULTS_DIR)
        os.makedirs(dir_path, exist_ok=True)

        with open(os.path.join(dir_path, "train_results.json"), "w+") as f:
            json.dump(results[mdl], f, indent=4)

        plt.figure()
        plt.plot(range(1, args.epochs + 1), results[mdl]["best"]["train_losses"])
        plt.plot(range(1, args.epochs + 1), results[mdl]["best"]["val_losses"])
        plt.xticks(range(1, args.epochs + 1))

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["Train Loss", "Val Loss"])
        plt.savefig(os.path.join(dir_path, "loss_curve.png"))

    # start testing
    for mdl in mdls:
        print(f"Starting Testing for {mdl}...")
        # load the best model across all hyperparameters
        data = torch.load(os.path.join(CHECKPOINT_DIR, f"{mdl}.pt"))
        model = get_model(mdl, dropout_rate, initializer).to(device)
        model.load_state_dict(data["model_state_dict"])

        test_metrics, test_loss, conf_matrix = test_model(
            model, criterion, test_loader, device
        )
        test_metrics["args"] = data["args"]

        dir_path = os.path.join(mdl, START_DATE, TEST_RESULTS_DIR)
        os.makedirs(dir_path, exist_ok=True)

        sns.heatmap(conf_matrix, annot=True)
        plt.savefig(os.path.join(dir_path, "confusion_matrix.png"))

        with open(os.path.join(dir_path, "test_results.json"), "w+") as f:
            json.dump(test_metrics, f, indent=4)
