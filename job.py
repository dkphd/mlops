import wandb
import torch
run = wandb.init(
    project="physionet_article",
    # name=f"resnet18_{'_'.join(new_config.folds_train)}_{train_config.activation}",
    name="test",
    entity="phd-dk",
)
        


run.log({"hello": "world"})
for i in range(10):
    run.log({"metric": i})

for i in range(10):
    if torch.cuda.is_available():
        run.log({"cuda": 1})
    else:
        run.log({"cuda": 0})




    
