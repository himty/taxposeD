import numpy as np
import torch
import pytorch_lightning as pl
from torchvision.transforms import ToTensor
import wandb
import torch.nn.functional as F

to_tensor = ToTensor()
torch.set_printoptions(threshold=5, linewidth=1000)

class PointCloudTrainingModule(pl.LightningModule):

    def __init__(self,
                 model=None,
                 lr=1e-3,
                 image_log_period=500):
        super().__init__()
        self.model = model
        self.lr = lr
        self.image_log_period = image_log_period
        self.global_val_step = 0
        self.automatic_optimization = True

    def module_step(self, batch, batch_idx):
        raise NotImplementedError(
            'module_step must be implemented by child class')
        return loss, log_values

    def visualize_results(self, batch, batch_idx):
        return {}

    def manual_training_step(self, batch, batch_idx):
        opt = self.optimizers()
        
        batch_size = batch[list(batch.keys())[0]].shape[0]
        loss, log_values = self.module_step(batch, batch_idx)

        for key, val in log_values.items():
            self.log(key, val, batch_size=batch_size)

        if((self.global_step % self.image_log_period) == 0):
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if(isinstance(val, wandb.Object3D) or isinstance(val, wandb.Html)):
                    wandb.log(
                        {key: val, "trainer/global_step": self.global_step, })
                else:
                    self.logger.log_image(
                        key, images=[val],  # self.global_step
                    )
        self.log('train_loss', loss, batch_size=batch_size, prog_bar=True)
        
        # opt.zero_grad()
        # self.manual_backward(loss)
        # opt.step()

        # # Print all the gradients
        # for name, param in self.named_parameters():
        #     print(f'param: {name} | grad: {param.grad}')
        
        # print(f'loss: {loss}')
        
        # breakpoint()
        
        return loss

    def training_step(self, batch, batch_idx):
        if not self.automatic_optimization:
            return self.manual_training_step(batch, batch_idx)
        
        batch_size = batch[list(batch.keys())[0]].shape[0]
        loss, log_values = self.module_step(batch, batch_idx)

        for key, val in log_values.items():
            self.log(key, val, batch_size=batch_size)

        if((self.global_step % self.image_log_period) == 0):
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if(isinstance(val, wandb.Object3D) or isinstance(val, wandb.Html)):
                    wandb.log(
                        {key: val, "trainer/global_step": self.global_step, })
                else:
                    self.logger.log_image(
                        key, images=[val],  # self.global_step
                    )
        self.log('train_loss', loss, batch_size=batch_size)
        
        # Temp way to disable training
        # loss = torch.zeros(loss.shape)
        # loss.requires_grad = True
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch_size = batch[list(batch.keys())[0]].shape[0]
        loss, log_values = self.module_step(batch, batch_idx)

        for key, val in log_values.items():
            self.log(f'val_{key}/val_{dataloader_idx}', val, add_dataloader_idx=False, batch_size=batch_size)

        if((self.global_val_step % self.image_log_period) == 0):
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if(isinstance(val, wandb.Object3D), isinstance(val, wandb.Html)):
                    wandb.log(
                        {f'val_{key}/val_{dataloader_idx}': val, "trainer/global_step": self.global_val_step, })
                else:
                    self.logger.log_image(
                        f'val_{key}/val_{dataloader_idx}', images=[val],  # self.global_val_step
                    )
        self.global_val_step += 1

        self.log(f'val_loss/val_{dataloader_idx}', loss, add_dataloader_idx=False, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        batch_size = batch[list(batch.keys())[0]].shape[0]
        loss, log_values = self.module_step(batch, batch_idx)

        for key, val in log_values.items():
            self.log(key, val, batch_size=batch_size)

        if((self.global_step % self.image_log_period) == 0):
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if(isinstance(val, wandb.Object3D), isinstance(val, wandb.Html)):
                    wandb.log({'test_' + key: val, })
                else:
                    self.logger.log_image(
                        'test_' + key, val)

        self.log('test_loss', loss, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        return optimizer
