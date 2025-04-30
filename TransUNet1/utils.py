import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            if i == 3:
                temp_prob = input_tensor == 4  # * torch.ones_like(input_tensor)
                tensor_list.append(temp_prob.unsqueeze(1))
                # i = 0,1,2
            else:
                temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
                tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)  # (8,9,224,224)
        target = self._one_hot_encoder(target)  # (8,224,224-->(8,9,224,224)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])  # 第4个通道存的是target=4的true/false
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1  # 对应位置进行dice/hd95
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()  # 没有batch了，变成了（148,512,512）
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):  # 每一个切片而言 这边3维变二维的
            slice = image[ind, :, :]  # 把图像中的每一张切片分出来给slice
            x, y = slice.shape[0], slice.shape[1]  # 都是原图像512,512
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0，变成了(224,224)
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()  # input = (1,1,224,224),因为要进网络训练
            net.eval()
            with torch.no_grad():
                outputs = net(input)  # (1,9,224,224)
                # torch.argmax是返回最大值得索引值，比如在0~8个通道（即类别中），哪个通道对应的值是最大的，就把该通道的索引值（即通道类别值）返回回来
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)  # （224,224），每一个像素值取其最大的？
                out = out.cpu().detach().numpy()  # (224,224)
                if x != patch_size[0] or y != patch_size[1]:  # 512 ！= 224
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)  # 又回到（512,512）了
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):  # 0为背景 1~3,注意3对应的是标签4 prediction=3, label==4
        print(np.where(prediction == i))
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    # save nii格式
    # if test_save_path is not None:
    #     img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    #     prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    #     lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    #     img_itk.SetSpacing((1, 1, z_spacing))
    #     prd_itk.SetSpacing((1, 1, z_spacing))
    #     lab_itk.SetSpacing((1, 1, z_spacing))
    #     sitk.WriteImage(prd_itk, test_save_path + '/' + "_pred.nii.gz")
    #     sitk.WriteImage(img_itk, test_save_path + '/'+  "_img.nii.gz")
    #     sitk.WriteImage(lab_itk, test_save_path + '/'+  "_gt.nii.gz")
    return metric_list
