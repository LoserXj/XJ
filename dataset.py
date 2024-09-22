import os 
from torch.utils.data import Dataset, DataLoader
import pickle 
import torch 
import torch.nn as nn
import torch.nn.functional as F


def crop_tensor_to_tiles(tensor_feature, tensor_label, patch_size=(256,256), stride=256):
    h, w = tensor_feature.shape
    ph, pw = patch_size
    # 计算填充的大小
    pad_h = max(0, h%patch_size[0])
    pad_w = max(0, w%patch_size[1])
    # 定义pad阈值
    threshold_x = int(patch_size[0]/3)
    threshold_y = int(patch_size[1]/3)
    if pad_h < threshold_x:
        pad_h = 0
    if pad_w < threshold_y:
        pad_w = 0
    if pad_h != 0:
        pad_h = patch_size[0] - pad_h
    if pad_w != 0:
        pad_w = patch_size[1] - pad_w
    padded_tensor_feature = F.pad(tensor_feature, (0, pad_w, 0, pad_h), mode='constant', value=0)
    padded_tensor_label = F.pad(tensor_label, (0, pad_w, 0, pad_h), mode='constant', value=0)
    padded_h, padded_w = padded_tensor_feature.shape
    patches_feature = []
    patches_label = []
    for i in range(0, padded_h - ph + 1, stride):
        for j in range(0, padded_w - pw + 1, stride):
            patch_feature = padded_tensor_feature[i:i + ph, j:j + pw]
            patch_label = padded_tensor_label[i:i + ph, j:j + pw]
            patches_feature.append(patch_feature.unsqueeze(0).unsqueeze(1))
            patches_label.append(patch_label.unsqueeze(0).unsqueeze(1))
    return torch.cat(patches_feature,dim=0), torch.cat(patches_label,dim=0)


class YinYingCNNDataset(Dataset):
    def __init__(self, root_dir, patch_size=256, transform=None):
        """
        Args:
            root_dir (string): 数据集的根目录路径。
            transform (callable, optional): 需要应用于样本的可选变换。
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.feature_path_list= []
        self.label_path_list = []
        self.angle_path_list = []
        self.feature_tensor = []
        self.label_tensor = []
        self.mask_tensor = []
        self._load_dataset_path()

    def _load_dataset_path(self):
        """
        遍历数据集目录，加载所有数据和标签的路径。
        """
        for class_folder in os.listdir(self.root_dir):
            class_folder_path = os.path.join(self.root_dir, class_folder)
            for sub_class_folder in os.listdir(class_folder_path):
                sub_class_folder_path = os.path.join(class_folder_path,sub_class_folder)
                feature_path = os.path.join(sub_class_folder_path,'train','train.pkl')
                label_path = os.path.join(sub_class_folder_path,'label','label.pkl')
                angle_path = os.path.join(sub_class_folder_path,'train','position.pkl')
                self.feature_path_list.append(feature_path)
                self.label_path_list.append(label_path)
                self.angle_path_list.append(angle_path)
        
        # 拆分数据集
        for idx in range(len(self.feature_path_list)):
            feature_path = self.feature_path_list[idx]
            label_path = self.label_path_list[idx]
            angle_path = self.angle_path_list[idx]
            feature = None 
            label = None 
            angle = None 
            with open(feature_path,'rb') as f:
                feature = pickle.load(f)
            with open(label_path,'rb') as f:
                label = pickle.load(f)
            with open(angle_path,'rb') as f:
                angle = pickle.load(f)
            feature = feature.squeeze(0)
            tmp_feature_list, tmp_label_list = crop_tensor_to_tiles(torch.Tensor(feature),torch.Tensor(label),(self.patch_size,self.patch_size))
            for i in range(len(tmp_feature_list)):
                tmp = tmp_feature_list[i]
                tmp_label = tmp_label_list[i]
                mask = (tmp!=0).float()
                if (mask.sum()/(mask.shape[0]*mask.shape[1])<0.05):
                    continue 
                self.label_tensor.append(tmp_label)
                self.mask_tensor.append(mask.squeeze(0))
                tmp1 = torch.full((1,self.patch_size,self.patch_size),angle[0])
                tmp2 = torch.full((1,self.patch_size,self.patch_size),angle[1])
                tmp = torch.cat((tmp,tmp1,tmp2),dim=0)
                self.feature_tensor.append(tmp)


    def __len__(self):
        return len(self.feature_tensor)

    def __getitem__(self, idx):
        feature = self.feature_tensor[idx]
        label = self.label_tensor[idx]
        mask = self.mask_tensor[idx]    
        return feature,label,mask 
