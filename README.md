# Sentiment analysis with unsupervised model

## Directory
    ├── checkpoint/ 
    ├── data/
    │   ├── train.json
    │   ├── val.json
    │   ├── test.json
    ├── utils/
    │   ├── dataloader.py
    │   ├── model.py
    ├── train.py
    └── test.py


## Train

For training, you can run commands like this:

```shell
python3 train.py --train_dataset ./data/train.json --val_dataset ./data/val.json --batch_size 256 --epochs 20 --hidden_size 400 --embedd_dim 300 --checkpoint ./checkpoint/ --lr 3e-3 --num_aspect 14 --patience 5 --delta 1e-6
```


## Test

For evaluation, the command may like this:

```shell
python3 test.py --train_dataset ./data/train.json --test_dataset ./data/test.json --checkpoint ./checkpoint/path_to_the_best_checkpoint
```