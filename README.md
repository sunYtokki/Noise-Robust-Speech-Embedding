### Example Usage
Run the following command to train the model:  
`python train_byol.py --config config/default_wavlm-large_byol.yaml --device cuda:1`  

Check `config/default_wavlm-large_byol.yaml` for additional configuration options. 

### Logging
To use wandb logging, you might need your wandb account.
Refer https://wandb.me/wandb-core for more information about wandb API. 
To disable wandb, run `export WANDB_MODE=disabled` in terminal or set `disabled` in your config file
```yaml
logging:
  wandb_mode: "disabled"
```


### Credits
Codes under `baseline/` directory is a fork of [msplabresearch/MSP-Podcast_Challenge](https://github.com/msplabresearch/MSP-Podcast_Challenge). 
BibTeX:
```
@InProceedings{Goncalves_2024,
  author={L. Goncalves and A. N. Salman and A. {Reddy Naini} and L. Moro-Velazquez and T. Thebaud and L. {Paola Garcia} and N. Dehak and B. Sisman and C. Busso},
  title={Odyssey2024 - Speech Emotion Recognition Challenge: Dataset, Baseline Framework, and Results},
  booktitle={Odyssey 2024: The Speaker and Language Recognition Workshop},
  volume={To appear},
  year={2024},
  month={June},
  address =  {Quebec, Canada},
}
```