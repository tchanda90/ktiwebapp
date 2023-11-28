import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

cv2.setNumThreads(1)
import pytorch_lightning as pl
import albumentations
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from albumentations.pytorch import ToTensorV2
from captum.attr import LayerAttribution, LayerGradCam
from torchmetrics import Accuracy, AUROC, Recall, Specificity
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle
import sys
import json
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from skimage import draw
from skimage import io
#from config import *

import random
import os
import numpy as np
import pandas as pd
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

img_dir = "/home/caduser/Tirtha/overlap/data/HAM10000/HAM10000"
annotations_dir = "/home/caduser/Tirtha/overlap/data/ground_truth/annotations_gt"
metadata_file = "/home/caduser/Tirtha/overlap/data/ground_truth/metadata_gt.csv"
#metadata_file = "metadata_undersampled.csv"


model_save_dir = "./models"

weighted_sampling = False
save_attention_plots = False


num_epochs = 30
learning_rate = 0.0001
batch_size = 32
image_size = 224

attention_weight = 10
char_weight = 1
dropout = 0.4

mel_class_labels = ['TRBL', 'BDG', 'WLSA', 'ESA', 'GP', 'PV', 'PRL']
nev_class_labels = ['OPC', 'SPC', 'PRLC', 'PLF', 'PDES']
labels_mapping = {
    'TRBL': 'Dicke retikuläre oder verzweigte Linien', 'BDG': 'Schwarze Punkte oder Schollen in der Läsionsperipherie',
    'WLSA': 'Weiße Linien oder weißes strukturloses Areal',
    'ESA': 'Exzentrisch gelegenes, strukturloses Areal jeglicher Farbe, außer hautfarben, weiß und grau',
    'GP': 'Graue Muster', 'PV': 'Polymorphe Gefäße', 'PRL': 'Pseudopodien oder radiale Linien am Läsionsrand, die nicht den gesamten Läsionsumfang einnehmen',
    'OPC': 'Nur ein Muster und nur eine Farbe', 'SPC': 'Symmetrische Kombination von Mustern und/oder Farben',
    'PRLC': 'Pseudopodien oder radiale Linien am Läsionsrand über den gesamten Läsionsumfang',
    'PLF': 'Parallele Linien in den Furchen', 'PDES': 'Pigmentierung überschreitet Narbe nicht (nur nach Entfernung)'
}
descriptions = {
    'TRBL': 'Dicke Linien bezeichnen Linien, die mindestens so breit sind wie die zwischen den Linien liegenden Bereiche.',
    'ESA': 'Strukturlose Zonen finden sich bei gutartigen Läsionen häufig in der Mitte. Wenn sie nicht zentral liegen, ist eine strukturlose Zone ein Hinweis auf Bösartigkeit.',
    'BDG': 'Schwarze Punkte und Klumpen entstehen durch Melaninansammlungen in der oberflächlichen Epidermis. Zentrale schwarze Strukturen treten bei Naevi auf, aber wenn sie sich in der Peripherie befinden, können sie auf eine pagetoide Ausbreitung von melaninbeladenen Melanozyten zurückzuführen sein, und es sollte ein Melanom vermutet werden.',
    'WLSA': 'Weiße Linien und weiße, strukturlose Bereiche sind Hinweise auf eine bösartige Erkrankung, und sie müssen deutlich weißer sein als die umgebende normale Haut.',
    'GP': 'Graue Linien, Kreise, Klumpen oder Punkte können durch Melanin in der Dermis verursacht werden. Dieses Melanin kann sich in neoplastischen Zellen oder in Melanophagen befinden. Wenn eine Läsion chaotisch ist, kann Melanin in der Dermis auf Malignität hindeuten.',
    'PV': 'Von polymorphen Gefäßen spricht man, wenn in einer Läsion mehr als ein Gefäßmuster zu sehen ist, was ein Hinweis auf Bösartigkeit ist. ',
    'PRL': 'Pseudopods sind knollige Ausstülpungen am Rand der Läsion und treten bei Läsionen auf, die ein schnelles horizontales Wachstum aufweisen. Wenn sie über den gesamten Umfang verteilt sind, ist eine gutartige Diagnose am wahrscheinlichsten, aber wenn sie nur an einigen Teilen des Randes auftreten, besteht der Verdacht auf ein Melanom.',
    'OPC': 'Zum Beispiel nur braune netzartige Linien oder nur hautfarbene Kügelchen',
    'SPC': 'Die Symmetrie wird anhand des Pigmentmusters (Farbe und/oder Struktur) beurteilt. Ein symmetrisches Pigmentmuster ist typisch für gutartige Hautläsionen.',
    'PRLC': 'Pseudopods sind knollige Ausstülpungen am Rand der Läsion und treten bei Läsionen auf, die ein schnelles horizontales Wachstum aufweisen. Wenn sie über den gesamten Umfang verteilt sind, ist eine gutartige Diagnose am wahrscheinlichsten, aber wenn sie nur an einigen Teilen des Randes auftreten, besteht der Verdacht auf ein Melanom.',
    'PLF': 'Relevant ist hierbei, dass die Furchen (oft nur am Läsionsrand erkennbar) unpigmentiert sind.'
}

char_class_labels = mel_class_labels+nev_class_labels
pos_weight = torch.tensor([2, 2, 2.2, 2.2, 2, 2.5, 2.5, 1, 1, 1, 1, 1])

thresholds = torch.tensor([-0.7158, -0.7768, -1.3213, -1.0486, -0.5801, -1.6449, -2.1001, -2.2971,
        -1.6637, -2.9522, -5.0077, -5.6553])

char_class_labels_pred = [label+'_pred' for label in char_class_labels]
mel_class_labels_pred = [label+'_pred' for label in mel_class_labels]
nev_class_labels_pred = [label+'_pred' for label in nev_class_labels]
char_class_labels_score = [label+'_score' for label in char_class_labels]

annotation_labels = [label+'_annotation' for label in char_class_labels]

dx_class_label = ['benign_malignant']

seed = 42

seed_everything(seed)

import pickle
import sys
import json
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from skimage import draw
from skimage import io


class MelanomaCharacteristicsDataset(Dataset):
    def __init__(self, img_dir, annotations_dir, metadata, index=None, transform=None):
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.metadata = metadata
        if index is not None:
            self.metadata = self.metadata.loc[index]
        self.y = self.metadata[char_class_labels].values.astype(int)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        image_id = self.metadata.iloc[index]['image_id']

        image = io.imread(os.path.join(self.img_dir, image_id + '.jpg'))

        y_dx = torch.tensor(self.metadata.iloc[index][dx_class_label]).float()
        y_char = torch.tensor(self.metadata.iloc[index][char_class_labels]).float()

        # Load the annotation json
        with open(os.path.join(self.annotations_dir, image_id + '.json'), 'r') as f:
            y_annotations = json.loads(json.load(f))

        # Store the feature masks in a list to pass to the augmentations function
        masks = []
        for char in char_class_labels:
            # Cast mask lists to np arrays. Assign zero valued masks to features not present in the image.
            y_annotations[char] = np.array(y_annotations[char]) if char in y_annotations else np.zeros(
                (image_size, image_size))
            masks.append(y_annotations[char])

        if self.transform:
            transformed = self.transform(image=image, masks=masks)

            image, y_annotations = transformed['image'], transformed['masks']

        y_annotations = torch.tensor(y_annotations).float()

        return image, (y_dx, y_char, y_annotations, image_id)


class HAM10000Dataset(Dataset):
    def __init__(self, root_dir, metadata, index=None, transform=None):
        self.root_dir = root_dir
        self.metadata = metadata
        if index is not None:
            self.metadata = self.metadata.loc[index]
        self.y = self.metadata[dx_class_label].values.flatten().astype(int)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.metadata.iloc[index]['image'])

        image = io.imread(img_path)
        y_dx = torch.tensor(self.metadata.iloc[index][dx_class_label]).float()
        y_char = torch.tensor(self.metadata.iloc[index][dx_class_label]).float()
        y_annotations = torch.zeros(1, image_size, image_size)
        image_name = self.metadata.iloc[index]['image']

        if self.transform:
            image = self.transform(image=image)['image']

        return image, (y_dx, y_char, y_annotations, image_name)


def get_transforms(image_size, full=False):
    if full:
        transforms_train = albumentations.Compose([
            albumentations.Transpose(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ColorJitter(p=0.5),
            albumentations.OneOf([
                albumentations.MotionBlur(blur_limit=5),
                albumentations.MedianBlur(blur_limit=5),
                albumentations.GaussianBlur(blur_limit=(3, 5)),
                albumentations.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),
            albumentations.OneOf([
                albumentations.OpticalDistortion(distort_limit=1.0),
                albumentations.GridDistortion(num_steps=5, distort_limit=1.),
                albumentations.ElasticTransform(alpha=3),
            ], p=0.7),
            albumentations.CLAHE(clip_limit=4.0, p=0.7),
            albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            albumentations.Resize(image_size, image_size),
            albumentations.CoarseDropout(max_height=int(image_size * 0.375), max_width=int(image_size * 0.375),
                                         max_holes=1, p=0.3),
            albumentations.Normalize(),
            ToTensorV2()
        ])
    else:
        transforms_train = albumentations.Compose([
            albumentations.Transpose(p=0.2),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ColorJitter(p=0.5),
            albumentations.CLAHE(clip_limit=4.0, p=0.7),
            albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            # albumentations.CoarseDropout(max_height=int(image_size * 0.375), max_width=int(image_size * 0.375), max_holes=1, p=0.3),
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(),
            ToTensorV2()
        ])

    transforms_test = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(),
        ToTensorV2()
    ])
    return transforms_train, transforms_test


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        num = targets.shape[0]
        inputs = inputs.reshape(num, -1)
        targets = targets.reshape(num, -1)

        intersection = (inputs * targets).sum(1)
        dice = (2. * intersection + smooth) / (inputs.sum(1) + targets.sum(1) + smooth)

        dice = dice.sum() / num

        return 1 - dice


class CharacteristicsClassifier(pl.LightningModule):
    def __init__(self, hidden_size=64, learning_rate=2e-4, img_dir='./', annotations_dir='./', metadata_file='./',
                 train_data_dir='./', val_data_dir='./',
                 test_data_dir='./', batch_size=32, weighted_sampling=False):

        super().__init__()
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.metadata_file = metadata_file

        self.weighted_sampling = weighted_sampling
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.attributions = []
        self.y_annotations = []
        self.y_char = []
        self.lossC_list = []
        self.lossA_list = []
        self.dataset = []
        self.masks = []

        self.train_set, self.val_set, self.test_set = None, None, None

        self.lossC = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.lossA = DiceLoss()  # nn.MSELoss()

        self.labels = char_class_labels
        self.num_classes = len(char_class_labels)

        self.train_transform, self.test_transform = get_transforms(image_size=image_size)

        resnet = models.resnet50(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=resnet.fc.in_features, out_features=self.num_classes)
            # nn.ReLU(),
            # nn.Dropout(p=dropout),
            # nn.Linear(in_features=50, out_features=self.num_classes)
        )
        self.base_model = resnet

        self.layer_gc = LayerGradCam(self.base_model, self.base_model.layer4[-1])

        self.sigmoid = nn.Sigmoid()

        self.target_layer = 'base_model.layer4'
        self.fmap = None
        self.grad = None
        self.handlers = []

        def save_fmaps(key):
            def hook(module, fmap_in, fmap_out):
                if isinstance(fmap_out, list):
                    fmap_out = fmap_out[-1]
                self.fmap = fmap_out

            return hook

        def save_grads(key):
            def hook(module, grad_in, grad_out):
                self.grad = grad_out[0]

            return hook

        target_module = dict(self.named_modules())[self.target_layer]
        self.handlers.append(
            target_module.register_forward_hook(save_fmaps(self.target_layer))
        )
        self.handlers.append(
            target_module.register_backward_hook(save_grads(self.target_layer))
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    def attribute(self, logits, class_idx):
        self.zero_grad()
        one_hot = F.one_hot((torch.ones(logits.size(0)) * class_idx).long(), self.num_classes).to(self.device)
        logits.backward(gradient=one_hot, retain_graph=True)
        weights = F.adaptive_avg_pool2d(self.grad, 1)
        attr = torch.mul(self.fmap, weights).sum(dim=1, keepdim=True)
        # attr = self.sigmoid(F.relu(attr)) * 2 - 1
        attr = F.relu(attr)
        return attr

    def forward(self, x, zero_non_predicted=True):

        output = self.base_model(x)

        # Get attributions for each class
        # attributions = torch.cat([self.layer_gc.attribute(x, class_idx, relu_attributions=True) for class_idx in range(self.num_classes)], dim=1)
        attributions = torch.cat([self.attribute(output, class_idx) for class_idx in range(self.num_classes)], dim=1)
        # Upsample the attributions to match the image size
        attributions = F.interpolate(attributions, size=(image_size, image_size), mode='nearest')

        predicted_classes = torch.round(self.sigmoid(output))
        # Set attributions of non predicted classes to zero by multiplying with the rounded prediction: 1 or 0.
        if zero_non_predicted:
            for batch_idx in range(attributions.size(0)):
                for class_idx in range(self.num_classes):
                    attributions[batch_idx][class_idx] *= predicted_classes[batch_idx][class_idx]

        return output, attributions

    def on_train_start(self):

        self.log("hp/attention_weight", float(attention_weight))
        self.log("hp/lr", learning_rate)
        self.log("hp/batch_size", float(batch_size))
        self.log("hp/dropout", dropout)
        self.log("hp/weighted_sampling", float(weighted_sampling))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        output, attributions = self(x)

        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight

        self.lossA_list.append(lossA.item())
        self.lossC_list.append(lossC.item())

        self.log("train/loss", loss, on_epoch=True, on_step=False)
        self.log("train/lossA", lossA, on_epoch=True, on_step=False)
        self.log("train/lossC", lossC, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y

        with torch.set_grad_enabled(True):
            output, attributions = self(x)

        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight

        self.log("val/loss", loss, on_epoch=True, on_step=False)
        self.log("val/lossA", lossA, on_epoch=True, on_step=False)
        self.log("val/lossC", lossC, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y

        with torch.set_grad_enabled(True):
            output, attributions = self(x)

        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight

        self.accuracy(output.round(), y_char.int())
        self.auroc(output, y_char.int())
        self.sensitivity(output, y_char.int())
        self.specificity(output, y_char.int())

    def on_test_epoch_end(self) -> None:
        self.log("test/bal_acc", self.accuracy.compute())
        self.log("test/auroc", self.auroc.compute())
        self.log("test/sensitivity", self.sensitivity.compute())
        self.log("test/specificity", self.specificity.compute())

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y

        with torch.set_grad_enabled(True):
            output, attributions = self(x)

        return output, attributions, y_char, image_name, self.inverse_normalize(x)

    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        if stage == 'fit':
            metadata = pd.read_csv(metadata_file)  # .drop_duplicates(subset='lesion_id', keep='last')

            test_set = metadata[metadata['split'] == 'test']
            train = metadata[metadata['split'] == 'train']

            # Drop lesion Ids from train set that are also in test set
            train = train[~train['lesion_id'].isin(test_set['lesion_id'])]

            # Drop rows where all labels are 0
            # train = train.loc[(train[char_class_labels] != 0).any(axis=1)]

            train_lesions, val_lesions = train_test_split(
                train.drop_duplicates('lesion_id')['lesion_id'], test_size=0.18,
                stratify=train.drop_duplicates('lesion_id')[dx_class_label],
                random_state=seed
            )

            train_set = train[train['lesion_id'].isin(train_lesions)]
            val_set = train[train['lesion_id'].isin(val_lesions)]

            self.train_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                            annotations_dir=self.annotations_dir,
                                                            metadata=train_set,
                                                            transform=self.train_transform)

            self.val_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                          annotations_dir=self.annotations_dir,
                                                          metadata=val_set,
                                                          transform=self.test_transform)

            self.test_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                           annotations_dir=self.annotations_dir,
                                                           metadata=test_set,
                                                           transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=10)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=10)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=10)












