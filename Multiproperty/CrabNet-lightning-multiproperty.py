import numpy as np
import pandas as pd
import collections
from collections import OrderedDict
import pytorch_lightning as L
import os
import re
import json
import tqdm

from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
# from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from torchmetrics.functional import mean_squared_error, mean_absolute_error

from pymatgen.core.composition import Composition
from crabnet.kingcrab import CrabNet_Multi

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, StepLR
from torch.optim.swa_utils import AveragedModel, SWALR

from utils.utils import (Lamb, Lookahead, RobustL1, BCEWithLogitsLoss,
                         EDMDataset, get_edm, Scaler, DummyScaler, CombinedScaler, count_parameters)
from utils.get_compute_device import get_compute_device
# from crabnet.utils.composition import _element_composition, get_sym_dict, parse_formula, CompositionError
#from utils.optim import SWA

data_type_np = np.float32
data_type_torch = torch.float32

import wandb


class CrabNetDataModule(L.LightningDataModule):
    def __init__(self, train_file: str , 
                 val_file: str, 
                 test_file: str,
                 n_elements ='infer', 
                 classification = False,
                 batch_size = 2**10,
                 predict_batch_size = 32,
                 scale = True,
                 pin_memory = True,
                 num_workers = 1):
        super().__init__()
        self.train_path = train_file
        self.val_path = val_file
        self.test_path = test_file
        self.batch_size = batch_size
        self.n_elements=n_elements
        self.pin_memory = pin_memory
        self.scale = scale
        self.predict_batch_size = predict_batch_size
        self.classification = classification
        self.num_workers = num_workers

    def prepare_data(self):
        ### loading and encoding trianing data
        if(re.search('.json', self.train_path )):
            self.data_train=pd.read_json(self.train_path)
        elif(re.search('.csv', self.train_path)):
            self.data_train=pd.read_csv(self.train_path)

        self.train_main_data = list(get_edm(self.data_train, elem_prop='mat2vec',
                                      n_elements=self.n_elements,
                                      inference=False,
                                      verbose=True,
                                      drop_unary=False,
                                      scale=self.scale))
        
        self.train_len_data = len(self.train_main_data[0])
        self.train_n_elements = self.train_main_data[0].shape[1]//2

        print(f'loading data with up to {self.train_n_elements:0.0f} '
              f'elements in the formula for training')
        
        ### loading and encoding validation data
        if(re.search('.json', self.val_path )):
            self.data_val=pd.read_json(self.val_path)
        elif(re.search('.csv', self.val_path)):
            self.data_val=pd.read_csv(self.val_path)
        
        self.val_main_data = list(get_edm(self.data_val, elem_prop='mat2vec',
                                      n_elements=self.n_elements,
                                      inference=True,
                                      verbose=True,
                                      drop_unary=False,
                                      scale=self.scale))
        
        self.val_len_data = len(self.val_main_data[0])
        self.val_n_elements = self.val_main_data[0].shape[1]//2

        print(f'loading data with up to {self.val_n_elements:0.0f} '
              f'elements in the formula for validation')
        
        ### loading and encoding testing data
        if(re.search('.json', self.test_path )):
            self.data_test=pd.read_json(self.test_path)
        elif(re.search('.csv', self.test_path)):
            self.data_test=pd.read_csv(self.test_path)
        
        self.test_main_data = list(get_edm(self.data_test, elem_prop='mat2vec',
                                      n_elements=self.n_elements,
                                      inference=True,
                                      verbose=True,
                                      drop_unary=False,
                                      scale=self.scale))
        
        self.test_len_data = len(self.test_main_data[0])
        self.test_n_elements = self.test_main_data[0].shape[1]//2

        print(f'loading data with up to {self.test_n_elements:0.0f} '
              f'elements in the formula for testing')

    def setup(self, stage: str):
        ### creating dataloaders for training
        if stage == "fit":
            self.train_dataset = EDMDataset(self.train_main_data, self.train_n_elements)
            self.val_dataset = EDMDataset(self.val_main_data, self.val_n_elements)
            
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = EDMDataset(self.test_main_data, self.test_n_elements)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          pin_memory=self.pin_memory, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                        pin_memory=self.pin_memory, shuffle=False,num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_len_data,
                        pin_memory=self.pin_memory, shuffle=False,num_workers=self.num_workers)


class CrabNetLightning(L.LightningModule):
    def __init__(self, **config):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = CrabNet_Multi(out_dims=config['out_dims'],
                             d_model=config['d_model'],
                             N=config['N'],
                             heads=config['heads'])
        print('\nModel architecture: out_dims, d_model, N, heads')
        print(f'{self.model.out_dims}, {self.model.d_model}, '
                  f'{self.model.N}, {self.model.heads}')
        print(f'Model size: {count_parameters(self.model)} parameters\n')

        ### here we define some important parameters
        self.fudge=config['fudge']
        self.batch_size=config['batch_size']
        self.classification = config['classification']
        self.base_lr=config['base_lr']
        self.max_lr=config['max_lr']
        ### here we also need to initialise scaler based on training data
        if(re.search('.json', config['train_path'] )):
            train_data=pd.read_json(config['train_path'])
        elif(re.search('.csv', config['train_path'])):
            train_data=pd.read_csv(config['train_path'])
        
        y=np.array([train_data['form_energy_per_atom'].values,train_data['stable'].values,
           train_data['energy_above_hull'].values, train_data['disorder']]) # flag
        
        self.step_size = len(train_data) # flag

        self.scaler=CombinedScaler(y,scale_map=[True,False,True,False])

        # maybe masked versions are needed
        self.criterion_energy = RobustL1
        self.criterion_stability = BCEWithLogitsLoss
        self.criterion_hull = RobustL1
        self.criterion_disorder = BCEWithLogitsLoss
        
        self.mask=np.isnan(y)
        ma_y=np.ma.array(y, mask=self.mask)
        self.weight_stability=torch.tensor(((len(y[1,:])-np.sum(y[1,:]))/np.sum(y[1,:]))) #was .cuda()
        self.weight_disorder=torch.tensor(((len(ma_y[3,:])-np.sum(ma_y[3,:].mask)-np.sum(ma_y[3,:]))/np.sum(ma_y[3,:]))) #was .cuda()


    def forward(self, src, frac):
        out=self.model(src, frac)
        return out

    def configure_optimizers(self):
        base_optim = Lamb(params=self.model.parameters(),lr=0.001)
        optimizer = Lookahead(base_optimizer=base_optim)
        lr_scheduler = CyclicLR(optimizer,
                                base_lr=self.base_lr,
                                max_lr=self.max_lr,
                                cycle_momentum=False,
                                step_size_up=self.step_size)
        # lr_scheduler=StepLR(optimizer,
        #                     step_size=3,
        #                     gamma=0.5)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        X, y, formula = batch
        y = self.scaler.scale(y)
        src, frac = X.squeeze(-1).chunk(2, dim=1)
        frac = frac * (1 + (torch.randn_like(frac))*self.fudge)
        frac = torch.clamp(frac, 0, 1)
        frac[src == 0] = 0
        frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
        
        output = self(src, frac)
        prediction=[]
        uncertainty=[]
        for i in range(4):
            pred, unc = output[i].chunk(2, dim=-1)
            prediction.append(pred)
            uncertainty.append(unc)

        loss = self.criterion_energy(prediction[0].view(-1),
                              uncertainty[0].view(-1),
                              y[:,0].view(-1))+\
               self.criterion_stability(prediction[1].view(-1),
                              uncertainty[1].view(-1),
                              y[:,1].view(-1),weight=self.weight_stability)+\
               self.criterion_hull(prediction[2].view(-1),
                              uncertainty[2].view(-1),
                              y[:,2].view(-1))+\
               self.criterion_disorder(prediction[3].view(-1),
                              uncertainty[3].view(-1),
                              y[:,3].view(-1), weight=self.weight_disorder)              
               
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        # uncertainty = torch.exp(uncertainty) * self.scaler.std
        # prediction = self.scaler.unscale(prediction)
        prediction[1] = torch.sigmoid(prediction[1])
        prediction[3] = torch.sigmoid(prediction[3])
        # if self.classification:
        #     prediction = torch.sigmoid(prediction)
        # if self.classification:
        y_pred = prediction[1].view(-1).detach().cpu().numpy() > 0.5
        acc=accuracy_score(y_pred,y[:,0].view(-1).detach().cpu().numpy())
        # f1=f1_score(y_pred,y.view(-1).detach().cpu().numpy())
        #     # auc=roc_auc_score(prediction.view(-1).detach().cpu().numpy(),y.view(-1).detach().cpu().numpy())
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        #     self.log("train_f1", f1, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        #     # self.log("train_auc", auc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        # else:
        #     mse = mean_squared_error(prediction.view(-1),y.view(-1))
        #     mae = mean_absolute_error(prediction.view(-1),y.view(-1))
        #     self.log("train_mse", mse, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        #     self.log("train_mae", mae, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y, formula = batch
        y = self.scaler.scale(y)
        src, frac = X.squeeze(-1).chunk(2, dim=1)
        frac = frac * (1 + (torch.randn_like(frac))*self.fudge)
        frac = torch.clamp(frac, 0, 1)
        frac[src == 0] = 0
        frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
        
        output = self(src, frac)
        prediction=[]
        uncertainty=[]
        for i in range(4):
            pred, unc = output[i].chunk(2, dim=-1)
            prediction.append(pred)
            uncertainty.append(unc)

        val_loss = self.criterion_energy(prediction[0].view(-1),
                              uncertainty[0].view(-1),
                              y[:,0].view(-1))+\
               self.criterion_stability(prediction[1].view(-1),
                              uncertainty[1].view(-1),
                              y[:,1].view(-1),weight=self.weight_stability)+\
               self.criterion_hull(prediction[2].view(-1),
                              uncertainty[2].view(-1),
                              y[:,2].view(-1))+\
               self.criterion_disorder(prediction[3].view(-1),
                              uncertainty[3].view(-1),
                              y[:,3].view(-1), weight=self.weight_disorder) 
        
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        # uncertainty = torch.exp(uncertainty) * self.scaler.std
        # prediction = self.scaler.unscale(prediction)
        prediction[1] = torch.sigmoid(prediction[1])
        prediction[3] = torch.sigmoid(prediction[3])
        # if self.classification:
        #     prediction = torch.sigmoid(prediction)
        # if self.classification:
        y_pred = prediction[1].view(-1).detach().cpu().numpy() > 0.5
        acc=accuracy_score(y_pred,y[:,0].view(-1).detach().cpu().numpy())
        # f1=f1_score(y_pred,y.view(-1).detach().cpu().numpy())
        #     # auc=roc_auc_score(prediction.view(-1).detach().cpu().numpy(),y.view(-1).detach().cpu().numpy())
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)


        # if self.classification:
        #     prediction = torch.sigmoid(prediction)
        # if self.classification:
        #     y_pred = prediction.view(-1).detach().cpu().numpy() > 0.5
        #     acc=accuracy_score(y_pred,y.view(-1).detach().cpu().numpy())
        #     f1=f1_score(y_pred,y.view(-1).detach().cpu().numpy())
        #     # auc=roc_auc_score(prediction.view(-1).detach().cpu().numpy(),y.view(-1).detach().cpu().numpy())
        #     self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        #     self.log("val_f1", f1, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        #     # self.log("val_auc", auc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        # else:
        #     mse = mean_squared_error(prediction.view(-1),y.view(-1))
        #     mae = mean_absolute_error(prediction.view(-1),y.view(-1))
        #     self.log("val_mse", mse, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        #     self.log("val_mae", mae, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        return val_loss
     
    def test_step(self, batch, batch_idx):
        X, y, formula = batch
        y = self.scaler.scale(y)
        src, frac = X.squeeze(-1).chunk(2, dim=1)
        frac = frac * (1 + (torch.randn_like(frac))*self.fudge)
        frac = torch.clamp(frac, 0, 1)
        frac[src == 0] = 0
        frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
        
        # self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        # if self.classification:
        #     prediction = torch.sigmoid(prediction)
        # if self.classification:
        #     y_pred = prediction.view(-1).detach().cpu().numpy() > 0.5
        #     acc=accuracy_score(y_pred,y.view(-1).detach().cpu().numpy())
        #     f1=f1_score(y_pred,y.view(-1).detach().cpu().numpy())
        #     # auc=roc_auc_score(prediction.view(-1).detach().cpu().numpy(),y.view(-1).detach().cpu().numpy())
        #     self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        #     self.log("test_f1", f1, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        #     # self.log("test_auc", auc, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        # else:
        #     mse = mean_squared_error(prediction.view(-1),y.view(-1))
        #     mae = mean_absolute_error(prediction.view(-1),y.view(-1))
        #     print(mse,mae)
        #     self.log("test_mse", mse, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        #     self.log("test_mae", mae, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
        return 
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X, y, formula = batch
        y = self.scaler.scale(y)
        src, frac = X.squeeze(-1).chunk(2, dim=1)
        frac = frac * (1 + (torch.randn_like(frac))*self.fudge)
        frac = torch.clamp(frac, 0, 1)
        frac[src == 0] = 0
        frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
        
        output = self(src, frac)
        prediction, uncertainty = output.chunk(2, dim=-1)
        uncertainty = torch.exp(uncertainty) * self.scaler.std
        prediction = self.scaler.unscale(prediction)
        if self.classification:
            prediction = torch.sigmoid(prediction)
        return formula, prediction, uncertainty

def main():
    pass
    

if __name__=='__main__':

    wandb.login(key='b11d318e434d456c201ef1d3c86a3c1ce31b98d7')

    print('Loading the data...')  

    path='/Users/elenapatyukova/Documents/GitHub/Disorder-prediction/Multiproperty/'

    config={'train_path': path+'data/materials_data/all_properties/train_total.csv',
            'val_path': path+'data/materials_data/all_properties/val_total.csv',
            'test_path': path+'data/materials_data/all_properties/test_total.csv',
            'out_dims': 3,
            'd_model': 512,
            'N': 3,
            'heads': 4,
            'classification': True,
            'batch_size': 2**12,
            'fudge': 0,
            'random_seed': 42,
            'swa_epoch_start' : 0.05,
            'swa_lrs': 1e-2,
            'base_lr': 5e-5,
            'max_lr': 6e-4,
            'schedule': 'CyclicLR',
            'patience': 10,
            'num_workers' : 1 }

    # config={'train_path': path+'data/materials_data/train.csv',
    #         'val_path': path+'data/materials_data/val.csv',
    #         'test_path': path+'data/materials_data/test.csv',
    #         'out_dims': 3,
    #         'd_model': 512,
    #         'N': 3,
    #         'heads': 4,
    #         'classification': True,
    #         'batch_size': 2**11,
    #         'fudge': 0,
    #         'random_seed': 11,
    #         'swa_epoch_start' : 0.1,
    #         'swa_lrs': 1e-3,
    #         'opt_lr': 1e-3,
    #         'step_size': 3,
    #         'gamma': 0.5,
    #         'schedule': 'StepLR',
    #         'patience': 10,
    #         'num_workers' : 1 }
    
    L.seed_everything(config['random_seed'])
    model = CrabNetLightning(**config)
   
    wandb_logger = WandbLogger(project="Crabnet-multiplroperty", config=config, log_model="all")

    # artifact1 = wandb.Artifact(name="train-set", type="dataset")
    # artifact1.add_file(local_path=config['train_path'])
    # wandb.log_artifact(artifact1)

    # artifact2 = wandb.Artifact(name="val-set", type="dataset")
    # artifact2.add_file(local_path=config['val_path'])
    # wandb_logger.log_artifact(artifact2)

    # artifact3 = wandb_logger.Artifact(name="test-set", type="dataset")
    # artifact3.add_file(local_path=config['test_path'])
    # wandb_logger.log_artifact(artifact3)

    trainer = Trainer(max_epochs=5,accelerator='mps', devices=1, logger=wandb_logger,
                      callbacks=[StochasticWeightAveraging(swa_epoch_start=config['swa_epoch_start'],swa_lrs=config['swa_lrs']),
                                EarlyStopping(monitor='train_loss', patience=config['patience']), ModelCheckpoint(monitor='val_loss', mode="min", 
                                dirpath=path+'models/trained_models/', filename='multi-{epoch:02d}-{val_acc:.2f}')])

    disorder_data = CrabNetDataModule(config['train_path'],
                                   config['val_path'],
                                   config['test_path'])
    
    trainer.fit(model, datamodule=disorder_data)

    trainer.test(ckpt_path='best',test_dataloaders=disorder_data.test_dataloader)


    wandb.finish()
    # print('Start sweeping with different parameters for RF...')

    # wandb.login(key='b11d318e434d456c201ef1d3c86a3c1ce31b98d7')

    # sweep_config = {
    # 'method': 'random',
    # 'parameters': {'n_estimators': {'values': [50, 100, 150, 200]},
    #                'class_weight': {'values':['balanced', 'balanced_subsample']},
    #                'criterion': {'values': ['gini', 'entropy', 'log_loss']}
    # }
    # }

    # sweep_id = wandb.sweep(sweep=sweep_config, project="RF-disorder-prediction-global-disorder")

    # wandb.agent(sweep_id, function=main, count=10)

    # wandb.finish()
