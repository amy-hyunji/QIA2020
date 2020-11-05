# QIA2020
code for QIA2020
https://www.kaggle.com/c/qia-hackathon2020

## Goal
- multimodal sentiment analysis model
- 7 labels (hap, ang, dis, fea, neu, sad, sur)
- dataset (train: 44370, val: 5386)
    - video
    - text embedding

## How to train
```python train.py```
- args setting will be saved as {args.name}.txt
- ckpt will be saved every epoch under {args.name} file

## How to test
```python test.py --ckpt-dir xxx.pt```
- check if the setting of args is same with training 
- this will save {args.name}.csv 
