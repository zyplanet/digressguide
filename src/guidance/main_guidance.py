import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import psi4
from rdkit import Chem
import torch
import wandb
import hydra
import os
import omegaconf
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import warnings


import src.utils as utils
from src.guidance.guidance_diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.datasets import qm9_dataset,zinc_dataset
from src.metrics.molecular_metrics import SamplingMolecularMetrics, TrainMolecularMetricsDiscrete
from src.analysis.visualization import MolecularVisualization
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.utils import update_config_with_new_keys
from src.guidance.qm9_regressor_discrete import Qm9RegressorDiscrete


warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    saved_cfg = cfg.copy()

    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    final_samples_to_generate = cfg.general.final_model_samples_to_generate
    final_chains_to_save = cfg.general.final_model_chains_to_save
    batch_size = cfg.train.batch_size
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg.general.final_model_samples_to_generate = final_samples_to_generate
    cfg.general.final_model_chains_to_save = final_chains_to_save
    cfg.train.batch_size = batch_size
    cfg = update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': 'guidance', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True),
              'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg

import sys
import pandas as pd
@hydra.main(version_base='1.1', config_path='./../../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    # assert dataset_config.name == "qm9", "Only QM9 dataset is supported for now"
    workdir = os.getcwd()
    print("os working dir",workdir)
    if "multirun" in workdir:
        home_prefix = "./../../../../"
    else:
        home_prefix = "./../../../"
    # sys.exit()
    if dataset_config.name == "zinc":
        datamodule = zinc_dataset.MosesDataModule(cfg)
        dataset_infos = zinc_dataset.MOSESinfos(datamodule,cfg)
        datamodule.prepare_data()
        train_smiles = pd.read_csv("/root/DiGress/src/datasets/zinc/raw/zinc_train.csv")["smiles"].tolist()
    else:
        datamodule = qm9_dataset.QM9DataModule(cfg, regressor=True)
        dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
        datamodule.prepare_data()
        train_smiles = qm9_dataset.get_train_smiles(cfg, datamodule, dataset_infos)

    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()
    
    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                            domain_features=domain_features)
    # if dataset_config.name == "zinc":
    #     dataset_infos.output_dims = {'X': 0, 'E': 0, 'y': 0}
    # else:
    #     dataset_infos.output_dims = {'X': 0, 'E': 0, 'y': 2 if cfg.general.guidance_target == 'both' else 1}
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)
    
    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features, 'load_model': True}

    # When testing, previous configuration is fully loaded
    guidance_sampling_model = DiscreteDenoisingDiffusion(cfg, **model_kwargs)
    # cfg_pretrained, guidance_sampling_model = get_resume(cfg, model_kwargs)
    
    # OmegaConf.set_struct(cfg, True)
    # with open_dict(cfg):
    #     cfg.model = cfg_pretrained.model
    # model_kwargs['load_model'] = False
    print("hahahah")
    utils.create_folders(cfg)
    # cfg = setup_wandb(cfg)

    # load pretrained regressor
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    guidance_model = Qm9RegressorDiscrete.load_from_checkpoint(os.path.join(cfg.general.trained_regressor_path))

    model_kwargs['guidance_model'] = guidance_model

    if cfg.general.name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      gpus=cfg.general.gpus if torch.cuda.is_available() else 0,
                      limit_test_batches=100,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      strategy='ddp' if cfg.general.gpus > 1 else None,        # TODO CHANGE with ray
                      enable_progress_bar=False,
                      logger=[],
                      )
    print("hahahahaahah")
    # add for conditional sampling
    model = guidance_sampling_model
    model.args = cfg
    model.guidance_model = guidance_model
    sd_dict = {"planar":home_prefix+"pretrained/planarpretrained.pt",
                    "sbm":home_prefix+"pretrained/sbmpretrained.pt",
                    "zinc":home_prefix+"pretrained/zincpretrained.pt",
                    "moses":home_prefix+"pretrained/mosespretrained.pt"}
    # sd_dict = {}
    print("load path is {}".format(sd_dict[cfg.dataset.name]))
    if dataset_config.name == "zinc":
        sd = torch.load(sd_dict[cfg.dataset.name])
        new_sd = {}
        for k,v in sd.items():
            # print(k,v.shape)
            if "model" in k:
                new_sd[k[6:]]=v
        model.model.load_state_dict(new_sd)
        model.model.cuda()
        print("load pretrained model")
    # sys.exit()
    print("load from check point")
    trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
