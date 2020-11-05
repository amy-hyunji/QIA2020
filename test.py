import torch
import torch.nn as nn
import argparse
import os
import pandas as pd

from torch.utils.data import DataLoader
from dataloader import DataSet
from models import FaceResNet, VisualLstm, LangLstm, MultiModel

"""
Saves .csv file to submit
"""


def test():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=str, required=True, help="directory to .pt")
    parser.add_argument("--data-dir", default="../qia2020", type=str)
    parser.add_argument("--name", default="default", type=str)
    parser.add_argument("-emb", default=256, type=int)
    parser.add_argument("-text-lstm-layer", default=3, type=int)
    parser.add_argument("-visual-lstm-layer", default=3, type=int)
    parser.add_argument("-learning-rate", default=0.001, type=float)

    args = parser.parse_args()

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    testset = DataSet(args, "test")
    test_dataloader = DataLoader(testset, batch_size=1, shuffle=False, drop_last=False)

    """
    Load Models from saved pt
    """
    face_resnet = FaceResNet(args)
    visual_lstm = VisualLstm(args)
    lang_lstm = LangLstm(args)
    multi = MultiModel(args)

    ckpt = torch.load(args.ckpt_dir)
    face_resnet.load_state_dict(ckpt["face_resnet"])
    face_resnet.to(device).eval()
    visual_lstm.load_state_dict(ckpt["visual_lstm"])
    visual_lstm.to(device).eval()
    lang_lstm.load_state_dict(ckpt["lang_lstm"])
    lang_lstm.to(device).eval()
    multi.load_state_dict(ckpt["multi"])
    multi.to(device).eval()

    answerdict = {"FileID": [], "Emotion": []}

    for idx, b_data in enumerate(test_dataloader):

        if idx % 500 == 0:
            print(f"*** Testing [{idx}/{len(test_dataloader)}]")

        text = b_data["text"].float().to(device)
        imglist = b_data["image"]
        filename = b_data["filename"]

        text = text.permute(1, 0, 2)
        text_emb = lang_lstm(text)

        imgsize = 160
        img = torch.stack(imglist, dim=0).to(device)
        seq_len = img.shape[0]
        bs = img.shape[1]
        img = img.view(seq_len * bs, 3, imgsize, imgsize)
        img_emb = face_resnet(img)
        img_emb = img_emb.view(seq_len, bs, -1)
        img_emb = visual_lstm(img_emb)

        # concate text embedding and image embedding channel wise
        _input = torch.cat((text_emb, img_emb), dim=1)
        output = multi(_input)
        pred = output.data.max(1, keepdim=True)[1]

        if pred == 0:
            emo = "hap"
        elif pred == 1:
            emo = "ang"
        elif pred == 2:
            emo = "dis"
        elif pred == 3:
            emo = "fea"
        elif pred == 4:
            emo = "neu"
        elif pred == 5:
            emo = "sad"
        elif pred == 6:
            emo = "sur"

        answerdict["FileID"].append(filename[0])
        answerdict["Emotion"].append(emo)

    print(f"Saving to {args.name}.csv!")
    # Done scanning through test dataset
    df = pd.DataFrame(answerdict)
    df.to_csv(f"{args.name}.csv", index=False)


if __name__ == "__main__":
    test()
