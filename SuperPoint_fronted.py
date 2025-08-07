# in the name of God
#
import os
import logging
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from util import descriptor_loss_sparse, labels2Dto3D_flattened, batch_descriptor_loss_sparse


class SuperPoint_fronted:
    """
    This class acts as the training agent for the SuperPoint part of the GLVL network.
    It is dynamically loaded by train.py and manages the model, optimizer, loss calculation,
    and checkpointing for the feature detector and descriptor.
    """
    
    def __init__(self, config, save_path, device):
        """
        Initializes the training agent.
        Args:
            config (dict): The configuration dictionary from the .yaml file.
            save_path (str): The directory where logs and checkpoints will be saved.
            device (str): The device to run on ('cuda' or 'cpu').
        """
        self.config = config
        self.save_path = save_path
        self.device = device
        self.model = None
        self.optimizer = None
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_path, "sp_logs"))
        logging.info(f"SuperPoint frontend initialized. Logs will be saved in {self.save_path}/sp_logs")
    
    def loadModel(self, model):
        """
        Receives the main GeoLocalizationNet model and sets up the optimizer
        for the SuperPointNet submodule only.
        """
        self.model = model
        
        # We access .module because the model is wrapped in nn.DataParallel
        superpoint_params = self.model.module.SuperPointNet.parameters()

        self.optimizer = optim.Adam(
            superpoint_params,
            lr=self.config['model']['learning_rate']
        )
    
    def _detector_loss(self, semi, labels_2D, valid_mask=None):
        """
        Calculates the detector loss.
        Args:
            semi (torch.Tensor): The output heatmap from the detector head [B, 65, Hc, Wc].
            labels_2D (torch.Tensor): The ground truth keypoint locations [B, 1, H, W].
            valid_mask (torch.Tensor, optional): A mask for valid regions.
        Returns:
            torch.Tensor: The calculated detector loss.
        """
        labels_3D = labels2Dto3D_flattened(labels_2D, cell_size=8)
        
        loss = F.cross_entropy(semi, labels_3D, reduction='none')
        
        if valid_mask is not None:
            valid_mask = F.interpolate(valid_mask.float(), size=loss.shape[1:])
            loss = loss * valid_mask
            
            num_valid_pixels = torch.sum(valid_mask).clamp(min=1)
            loss = torch.sum(loss) / num_valid_pixels
        else:
            loss = torch.mean(loss)
        
        return loss
    
    def train_val_sample(self, sample, n_iter, train=True):
        """
        Processes a single batch of data for either training or validation.
        """
        if self.model is None or self.optimizer is None:
            raise RuntimeError("loadModel() must be called before training.")
        
        if train:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
        
        # Move all tensors in the sample to the correct device
        for key, tensor in sample.items():
            if isinstance(tensor, torch.Tensor):
                sample[key] = tensor.to(self.device)
        
        # Forward pass for the original and warped images
        outputs = self.model(sample['image'], 'SuperPoint')
        warped_outputs = self.model(sample['warped_img'], 'SuperPoint')
        
        # --- Calculate Detector Loss ---
        detector_loss = self._detector_loss(
            warped_outputs['semi'],
            sample['warped_labels'],
            sample.get('warped_valid_mask')
        )
        
        # --- Calculate Descriptor Loss using the BATCH-AWARE function ---
        descriptor_loss, _, pos_loss, neg_loss = batch_descriptor_loss_sparse(
            outputs['desc'],
            warped_outputs['desc'],
            sample['homographies'],
            device=self.device,
            **self.config['model']['sparse_loss']['params']
        )
        
        # Combine the losses
        total_loss = detector_loss + self.config['model']['sparse_loss']['params']['lamda_d'] * descriptor_loss
        
        # --- Backpropagation and Optimization ---
        if train:
            total_loss.backward()
            self.optimizer.step()
        
        # --- Logging to TensorBoard ---
        if n_iter % self.config['tensorboard_interval'] == 0:
            mode = 'train' if train else 'val'
            self.writer.add_scalar(f'SuperPoint/{mode}/detector_loss', detector_loss.item(), n_iter)
            self.writer.add_scalar(f'SuperPoint/{mode}/descriptor_loss', descriptor_loss.item(), n_iter)
            self.writer.add_scalar(f'SuperPoint/{mode}/positive_descriptor_loss', pos_loss.item(), n_iter)
            self.writer.add_scalar(f'SuperPoint/{mode}/negative_descriptor_loss', neg_loss.item(), n_iter)
            self.writer.add_scalar(f'SuperPoint/{mode}/total_loss', total_loss.item(), n_iter)
        
        return total_loss.item()
    
    def saveModel(self, n_iter):
        """
        Saves a checkpoint of the model and optimizer state.
        """
        checkpoint_dir = os.path.join(self.save_path, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'superPointNet_{n_iter}_checkpoint.pth.tar')
        
        torch.save({
            'n_iter': n_iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        
        logging.info(f"Saved SuperPoint checkpoint to {checkpoint_path}")
    

#cloner174