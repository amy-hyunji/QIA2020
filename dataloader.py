import os
import cv2
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class DataSet(Dataset):
    def __init__(self, args, mode):
        super(DataSet).__init__()
        self.mode = mode
        self.dict = {"text": [], "label": [], "image": [], "filename": []}
        self.imgsize = 160

        if mode == "train_val":
            # use both train and val for final model
            text_train = os.path.join(args.data_dir, "text_train.npy")
            text_val = os.path.join(args.data_dir, "text_val.npy")
            label_train = os.path.join(args.data_dir, "label_train.npy")
            label_val = os.path.join(args.data_dir, "label_val.npy")
            image_train = os.path.join(args.data_dir, "image_frame_crop_train")
            image_val = os.path.join(args.data_dir, "image_frame_crop_val")

            self.textlist = np.concatenate(np.load(text_train), np.load(text_val))
            self.labellist = np.concatenate(np.load(label_train), np.load(label_val))
            train_imagelist = os.listdir(image_train)
            val_imagelist = os.listdir(image_val)
            train_imagelist.sort()
            val_imagelist.sort()
            train_num = len(train_imagelist)
            self.imagelist = train_imagelist + val_imagelist

        elif mode == "test":
            # no label set for test dataset
            text = os.path.join(args.data_dir, f"text_{mode}".npy)
            image = os.path.join(args.data_dir, f"image_frame_crop_{mode}")

            self.textlist = np.load(text)
            self.imagelist = os.listdir(image)
            self.imagelist.sort()

        else:
            text = os.path.join(args.data_dir, f"text_{mode}".npy)
            label = os.path.join(args.data_dir, f"label_{mode}".npy)
            image = os.path.join(args.data_dir, f"image_frame_crop_{mode}")

            self.textlist = np.load(text)
            self.labellist = np.load(label)
            self.imagelist = os.listdir(image)
            self.imagelist.sort()

        if "train" in mode:
            _transform = [
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.125),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]

        else:
            _transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]

        self.transform = transforms.Compose(_transform)

        if mode == "train_val":
            # iter for train + val
            for i, file in enumerate(self.imagelist):
                text = self.textlist[i]
                self.dict["text"].append(text)
                label = self.labellist[i]
                self.dict["label"].append(label)
                self.dict["filename"].append(file)
                if i < train_num:
                    filepath = os.path.join(image_train, file)
                else:
                    filepath = os.path.join(image_val, file)
                imglist = []
                for img in os.listdir(filepath):
                    imglist.append(os.path.join(filepath, img))
                self.dict["image"].append(imglist)

        elif mode == "test":
            for i, file in enumerate(self.imagelist):
                text = self.textlist[i]
                self.dict["text"].append(text)
                self.dict["filename"].append(file)
                filepath = os.path.join(image, file)
                imglist = []
                for img in os.listdir(filepath):
                    imglist.append(os.path.join(filepath, img))
                self.dict["image"].append(imglist)

        else:
            for i, file in enumerate(self.imagelist):
                text = self.textlist[i]
                self.dict["text"].append(text)
                label = self.labellist[i]
                self.dict["label"].append(label)
                self.dict["filename"].append(file)
                filepath = os.path.join(image, file)
                imglist = []
                for img in os.listdir(filepath):
                    imglist.append(os.path.join(filepath, img))
                self.dict["image"].append(imglist)

    def __len__(self):
        return len(self.dict["text"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imglist = self.dict["image"][idx]
        _imglist = []

        for cnt, imgdir in enumerate(imglist):
            if cnt > 4:
                break
            img = Image.open(imgdir).convert("RGB")
            img = self.transform(img)
            _imglist.append(img)

        # add padding if len is smaller than 5
        if len(_imglist) < 5:
            addn = 5 - len(_imglist)
            for _ in range(addn):
                img = torch.zeros(3, self.imgsize, self.imgsize)
                _imglist.append(img)

        assert len(_imglist) == 5

        if self.mode == "test":
            return {
                "text": torch.tensor(self.dict["text"][idx]),
                "image": _imglist,
                "filename": self.dict["filename"][idx],
            }

        return {
            "text": torch.tensor(self.dict["text"][idx]),
            "label": torch.tensor([self.dict["label"][idx]]),
            "image": _imglist,
        }
