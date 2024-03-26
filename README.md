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

下载pretrained model: https://drive.google.com/file/d/1JGCKzh8KSLPyHk4gZS2TKgm8sdi7JBk4/view?usp=drive_link

下载后放到./src/pretrained/zincpretrained.pt

在outputs的checkpoints中选择训练好的regressor，然后更改./configs/experiment/guidance_zinc.yaml，将trained_regressor_path更改为对应regressor的绝对路径

将 ./src/run_guidance.sh中第二行的target替换成regressor预测的蛋白质

```
conda activate digress
cd src
bash run_guidance.sh
```

开始生成，生成结束后，分子会保存到./src/下面，以gen_smile开头
