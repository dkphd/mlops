import wandb
import torch
settings = wandb.Settings(disable_git=True)

with wandb.init(settings=settings) as run:
    run.log({"hello": "world"})
    for i in range(10):
        run.log({"metric": i})
    
    for i in range(10):
        if torch.cuda.is_available():
            run.log({"cuda": 1})
        else:
            run.log({"cuda": 0})
