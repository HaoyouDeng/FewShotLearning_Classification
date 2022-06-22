# Few-Shot Learning for Classification

Haoyou Deng (haoyoudeng@gmail.com)

An awesome implement for few-shot learning, including some classical methods (such as baseline, MatchingNet, ProtoNet, RelationNet and MAML) and cross-domain few-shot learning.

## Datasets
For few-shot learning and cross-domain few-shot learning, we use datasets: miniImageNet, CUB, Cars, Places, Plantae, CropDiseases, EuroSAT, ISIC and ChestX.

For downloading and using datasets, refer to [CDFSL-ATA](https://github.com/Haoqing-Wang/CDFSL-ATA).

## Pretrain
To get a pretrain backbone model.
```
python train.py --dataset miniImagenet --model [backbone] --method baseline --train_aug --not_warmup --name pretrain_[backbone] -g [GPU]
```
## Single domain
- Baseline( pretrain & fine-tune)
```
# Train
# Test
```
- Meta-learning method(MatchingNet, ProtoNet and MAML)
```
# Train
# Test
```

## Cross domain
- Baseline
```
# Train
python train.py --dataset miniImagenet --method baseline --name baseline --train_aug

# Test
python test.py --testset [target_dataset] --method baseline --name baseline --train_aug
```
- Baseline++
```
# Train
python train.py --dataset miniImagenet --method baseline++ --name baseline++ --train_aug

# Test
python test.py --testset [target_dataset] --method baseline --name baseline++ --train_aug
```

- Meta-learning method (MatchingNet, ProtoNet, RelationNet and MAML)
```
# Train
python train.py --model ResNet10 --method [method] --n_shot 1 --name [method]_1s --train_aug --num_workers 8 -g 0 --tag [method]_baseline

# Test
python test.py --model ResNet10 --method [method] --n_shot 1 --name [method]_1s --num_workers 8 -g 0 --tag [method]_baseline 
```

## Note
- This code is built upon the implementation from [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot) and [CDFSL-ATA](https://github.com/Haoqing-Wang/CDFSL-ATA).