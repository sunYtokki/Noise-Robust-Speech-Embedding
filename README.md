Example usage
`python train.py --config config/dev.yaml --device cuda:1`

To use wandb logging, you might need your wandb account.
Refer https://wandb.me/wandb-core for more information about wandb API. 
To disable wandb, run `export WANDB_MODE=offline` in terminal or set `offline` in config file
```yaml
logging:
  wandb_mode: "offline"
```