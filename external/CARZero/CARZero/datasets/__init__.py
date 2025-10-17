from . import data_module, pretraining_dataset

DATA_MODULES = {
    "pretrain": data_module.PretrainingDataModule,
    "pretrain_xh": data_module.PretrainingXHDataModule,
}
