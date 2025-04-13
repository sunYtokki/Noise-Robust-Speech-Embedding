### Example Usage
Run the following command to train the model:  
`python train.py --config config/dev.yaml --device cuda:1`  

Be sure to check `config/default.yaml` for additional configuration options. I recommend copying this default config file as your starting point.  
Also, ensure that the `checkpoint_dir` and `log_dir` paths are valid and exist.

### Logging
To use wandb logging, you might need your wandb account.
Refer https://wandb.me/wandb-core for more information about wandb API. 
To disable wandb, run `export WANDB_MODE=disabled` in terminal or set `disabled` in config file
```yaml
logging:
  wandb_mode: "disabled"
```