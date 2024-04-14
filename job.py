import random
import wandb


def run_training_run(epochs, lr):
    settings = wandb.Settings(job_source="artifact")
    run = wandb.init(
        project="launch_demo",
        job_type="eval",
        settings=settings,
        entity="phd-dk",
        # Simulate tracking hyperparameters
        config={
            "learning_rate": lr,
            "epochs": epochs,
        },
    )

    offset = random.random() / 5
    print(f"lr: {lr}")

    for epoch in range(2, epochs):
        # simulating a training run
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset
        wandb.log({"acc": acc, "loss": loss})

    run.log_code()
    run.finish()


run_training_run(epochs=10, lr=0.01)