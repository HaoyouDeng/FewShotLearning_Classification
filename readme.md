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
## Train & Test
- Baseline or Baseline++ (pretrain & fine-tune)
```
# Train
python train.py --dataset miniImagenet --model [backbone] --method baseline --name baseline --train_aug -g [GPU]

python train.py --dataset miniImagenet --model [backbone] --method baseline++ --name baseline++ --train_aug -g [GPU]

# Test
python test.py --testset [target_dataset] --model [backbone] --method baseline --n_shot [1/5] --name baseline --train_aug -g [GPU]

python test.py --testset [target_dataset] --model [backbone] --method baseline++ --n_shot [1/5] --name baseline --train_aug -g [GPU]
```

- Meta-learning method(MatchingNet, ProtoNet and MAML)
```
# Train
python train.py --model [backbone] --method [method] --n_shot 1 --name [method]_1s --train_aug -g [GPU]

python train.py --model [backbone] --method [method] --n_shot 5 --name [method]_5s --train_aug -g [GPU]

# Test
python test.py --testset [target_dataset] --model [backbone] --method [method] --n_shot 1 --name [method]_1s -g [GPU]

python test.py --testset [target_dataset] --model [backbone] --method [method] --n_shot 5 --name [method]_5s -g [GPU]
```

## Note
- This code is built upon the implementation from [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot) and [CDFSL-ATA](https://github.com/Haoqing-Wang/CDFSL-ATA).