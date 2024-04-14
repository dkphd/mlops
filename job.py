import wandb
import torch
settings = wandb.Settings(disable_git=True)

with wandb.init(settings=settings) as run:
    run.log({"hello": "world"})
    for i in range(10):
        run.log({"metric": i})
    run.log(
        {
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",

        }
    )
