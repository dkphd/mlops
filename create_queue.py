import json
import wandb
config ={"label":"tutorial-test"}
api = wandb.Api()
queue = api.create_run_queue('my_queue', 'local-container', prioritization_mode='V0', entity='phd-dk', config=config)