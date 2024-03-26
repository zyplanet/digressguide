# Denoising diffusion models for graph generation


## dependence
```
conda create --name digress --file spec-list.txt
conda activate digress
pip install -r requirements.txt
pip install -e .
```

## dataset
zinc: https://drive.google.com/file/d/1R9frhkoKKlhInVxzpOPD5YQmnhsxYfET/view?usp=drive_link

解压后放到./src/datasets/zinc

更改./configs/dataset/zinc.yaml，将datadir替换成./src/datasets/zinc的绝对路径

## train regressor

```
cd src
python guidance/train_qm9_regressor.py general.target="parp1"
```
更改target，有五种蛋白质："parp1","fa7","5ht1b","braf","jak2"

训练对应的五个regressor

## guidance generation

todo