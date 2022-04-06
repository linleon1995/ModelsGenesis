import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import unet3d
from config import models_genesis_config
import build_dataset
from utils import create_training_path


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = models_genesis_config()
config.display()

#Declare the Dice Loss
def torch_dice_coef_loss(y_true, y_pred, smooth=1.):
    # y_true_f = torch.flatten(y_true)
    # y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true * y_pred)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth))



def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = torch.reshape(input, [input.size()[0], -1])
    target = torch.reshape(target, [target.size()[0], -1])
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    # denominator = (input * input).sum(-1) + (target * target).sum(-1)
    # ii, tt = input.sum(-1), target.sum(-1)
    denominator = input.sum(-1) + target.sum(-1)
    # dsc = 2 * (intersect / denominator.clamp(min=epsilon))
    dsc = (2*intersect + epsilon) / (denominator + epsilon)
    # print(dsc)
    return torch.mean(dsc)


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


# prepare your own data

# # TODO: annotation json and cv
# train_cases = data_utils.get_pids_from_coco(
#     [os.path.join(cfg.DATA.NAMES[dataset_name]['COCO_PATH'], f'annotations_train.json') for dataset_name in cfg.DATA.NAMES])
# # valid_cases = data_utils.get_pids_from_coco(
#     [os.path.join(cfg.DATA.NAMES[dataset_name]['COCO_PATH'], f'annotations_test.json') for dataset_name in cfg.DATA.NAMES])

train_cases = [f'1m{idx:04d}' for idx in range(1, 37)] + [f'1B{idx:04d}' for idx in range(1, 21)]

# TODO: try to use dataset config
# key = '64x64x32-100x100x0-1.0' # 64x64x32-100x100x0-0.5
# input_roots = [os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\TMH-Malignant', key, 'Image'),
#                os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\TMH-Malignant', key, 'Image'),
#             #    os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\TMH-Benign', key, 'Image'),
#             #    os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\TMH-Benign', key, 'Image'),
#                ]
# target_roots = [os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\TMH-Malignant', key, 'Mask'),
#                 os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\TMH-Malignant', key, 'Mask'),
#                 # os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\TMH-Benign', key, 'Mask'),
#                 # os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\TMH-Benign', key, 'Mask'),
#                 ]

key = '32x64x64-10-shift-8'
input_roots = [
            os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'positive', 'Image'),
            os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'positive', 'Image'),
        #    os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'positive', 'Image'),
        #    os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'positive', 'Image'),
            os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'negative', 'Image'),
            os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'negative', 'Image'),
        #    os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'negative', 'Image'),
        #    os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'negative', 'Image'),
               ]
target_roots = [
            os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'positive', 'Mask'),
            os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'positive', 'Mask'),
            # os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'positive', 'Mask'),
            # os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'positive', 'Mask'),
            os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'negative', 'Mask'),
            os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'negative', 'Mask'),
            # os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'negative', 'Mask'),
            # os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'negative', 'Mask'),
                ]

train_loader, _ = build_dataset.build_dataloader(input_roots, target_roots, train_cases, train_batch_size=config.batch_size)


# Model Genesis provided LUNA16
# train_loader, valid_loader, train_samples, valid_samples = build_generate_pair(config, seg_model=True)

# prepare the 3D model
model = unet3d.UNet3D()

#Load pre-trained weights

if config.weights is not None:
    checkpoint = torch.load(config.weights)
    state_dict = checkpoint['state_dict']
    # state_dict = checkpoint['net']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    model.load_state_dict(unParalled_state_dict)

model.to(device)
model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
# criterion = torch_dice_coef_loss
criterion = DiceLoss(normalization='none')
# optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=0.9, weight_decay=0.0, nesterov=False)
optimizer = torch.optim.Adam(model.parameters(), config.lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config.patience * 0.8), gamma=0.5)
intial_epoch =0
checkpoint_saving_steps = 10
checkpoint_path = os.path.join(config.model_path, config.exp_name)
run_path = create_training_path(checkpoint_path)

# trainer = Trainer(model,
#                   criterion=criterion,
#                   optimizer=optimizer,
#                   train_dataloader=train_loader,
#                   valid_dataloader=None,
#                   logger=logger,
#                   device=configuration.get_device(),
#                   n_class=config.nb_class,
#                   exp_path=exp_path,
#                   train_epoch=config.nb_epoch,
#                   batch_size=config.batch_size,
#                   valid_activation=valid_activation,
#                   history=checkpoint_path,
#                   checkpoint_saving_steps=checkpoint_saving_steps)

# train the model
print('Start training')
min_loss = 10000
for epoch in range(intial_epoch, config.nb_epoch):
    # scheduler.step(epoch)
    model.train()
    losses = []
    for batch_ndx, (x, y) in enumerate(train_loader):
        # print(torch.max(x))
        # x, y = torch.from_numpy(x), torch.from_numpy(y)
        x, y = x.float().to(device), y.float().to(device)
        
        pred = model(x)

        # loss = criterion(y, pred, smooth=1e-5)
        loss = criterion(pred, y)

        x_np, y_np, pred_np = x.cpu().detach().numpy(), y.cpu().detach().numpy(), pred.cpu().detach().numpy()
        pred_np = np.where(pred_np>0.5, 1, 0)
        if batch_ndx%50 == 0:
            for n in range(6):
                for s in range(0, 32):
                    if np.sum(y_np[n,0,...,s]) > 0:
                        # print(np.sum(y_np[n,0,...,s]))
                        plt.imshow(x_np[n,0,...,s], 'gray')
                        plt.imshow(y_np[n,0,...,s]+2*pred_np[n,0,...,s], alpha=0.2, vmin=0, vmax=3)
                        plt.title(f'n: {n} s: {s}')
                        plt.savefig(f'figures/plot/train-{epoch}-{batch_ndx}-{n}-{s}.png')
                        # plt.show()

        if batch_ndx%200 == 0:
            print(f'Step {batch_ndx} Loss {loss}')
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = sum(losses)/len(losses)
    print(20*'-')
    print(f'Epoch {epoch} Loss {avg_loss}')


    checkpoint = {'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epoch': epoch
                  }

    if avg_loss < min_loss:
        min_loss = avg_loss
        torch.save(checkpoint, os.path.join(run_path, f'ckpt-best.pt'))

    if epoch%checkpoint_saving_steps == 0:
        torch.save(checkpoint, os.path.join(run_path, f'ckpt-{epoch:03d}.pt'))

        