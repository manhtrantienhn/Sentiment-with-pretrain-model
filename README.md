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
!python train.py --train_dataset ./data/t.json --val_dataset ./data/t.json --batch_size 64 --epochs 10 --hidden_size 400 --embedd_dim 300 --checkpoint ./checkpoint/ --lr 1e-4 --num_aspect 14 --patience 5 --delta 1e-6
```


## Test

For evaluation, the command may like this:

```shell
python3 test.py --train_dataset ./data/train.json --test_dataset ./data/test.json --checkpoint ./checkpoint/path_to_the_best_checkpoint
```