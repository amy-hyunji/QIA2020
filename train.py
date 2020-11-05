import torch
import torch.nn as nn
import argparse
import os
import json
from tqdm import tqdm
from models import FaceResNet, VisualLstm, LangLstm, MultiModel
from dataloader import DataSet
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../qia2020", type=str)
    parser.add_argument("--name", default="default", type=str)
    parser.add_argument("-emb", default=256, type=int)
    parser.add_argument("-batch_size", default=32, type=int)
    parser.add_argument("-epoch", default=10, type=int)
    parser.add_argument("-text-lstm-layer", default=3, type=int)
    parser.add_argument("-visual-lstm-layer", default=3, type=int)
    parser.add_argument("-learning-rate", default=0.001, type=float)

    args = parser.parse_args()

    # dump args to txt file
    with open(f"{args.name}.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    trainset = DataSet(args, "train")
    valset = DataSet(args, "val")

    train_dataloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_dataloader = DataLoader(
        valset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    """
    Load Models
    """
    face_resnet = FaceResNet(args).to(device)
    visual_lstm = VisualLstm(args).to(device)
    lang_lstm = LangLstm(args).to(device)
    multi = MultiModel(args).to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # optimizer & learning_rate
    parameters = (
        list(face_resnet.parameters())
        + list(visual_lstm.parameters())
        + list(lang_lstm.parameters())
        + list(multi.parameters())
    )
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=0.96
    )

    # make save file
    if not os.path.exists(args.name):
        os.makedirs(args.name)

    """
    Start Training
    """
    print("#" * 20)
    print("Start Training!")
    print("#" * 20)

    for epoch in tqdm(range(args.epoch)):
        face_resnet.train()
        visual_lstm.train()
        lang_lstm.train()
        multi.train()

        for b_data in train_dataloader:

            text = b_data["text"].float().to(device)
            label = b_data["label"].to(device)
            imglist = b_data["image"]

            optimizer.zero_grad()

            text = text.permute(1, 0, 2)
            text_emb = lang_lstm(text)

            imgsize = 160
            img = torch.stack(imglist, dim=0).to(
                device
            )  # [seq_len, bs, 3, imgsize, imgsize]
            seq_len = img.shape[0]
            bs = img.shape[1]
            img = img.view(seq_len * bs, 3, imgsize, imgsize)
            img_emb = face_resnet(img)
            img_emb = img_emb.view(seq_len, bs, -1)
            img_emb = visual_lstm(img_emb)

            # concat text embedding and image embedding channel wise
            _input = torch.cat((text_emb, img_emb), dim=1)
            output = multi(_input)

            # calculate loss & step
            loss = criterion(output, label.squeeze())
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Loss: {loss}")

        print("Saving...")
        torch.save(
            {
                "face_resnet": face_resnet.state_dict(),
                "lang_lstm": lang_lstm.state_dict(),
                "visual_lstm": visual_lstm.state_dict(),
                "multi": multi.state_dict(),
                "args": args,
            },
            os.path.join(args.name, f"{str(epoch)}.pt"),
        )

        # schedule learning rate
        lr_scheduler.step()

        face_resnet.eval()
        visual_lstm.eval()
        lang_lstm.eval()
        multi.eval()
        correct_num = 0

        with torch.no_grad():
            for b_data in val_dataloader:
                text = b_data["text"].float().to(device)
                label = b_data["label"].to(device)
                imglist = b_data["image"]

                text = text.permute(1, 0, 2)
                text_emb = lang_lstm(text)

                imgsize = 160
                img = torch.stack(imglist, dim=0).to(
                    device
                )  # [seq_len, bs, 3, imgsize, imgsize]
                seq_len = img.shape[0]
                bs = img.shape[1]
                img = img.view(seq_len * bs, 3, imgsize, imgsize)
                img_emb = face_resnet(img)
                img_emb = img_emb.view(seq_len, bs, -1)
                img_emb = visual_lstm(img_emb)

                # concat text embedding and image embedding channel wise
                _input = torch.cat((text_emb, img_emb), dim=1)
                output = multi(_input)
                pred = output.data.max(1, keepdim=True)[1]
                correct_num += pred.eq(label.data.view_as(pred)).cpu().sum().item()

            acc = correct_num / len(val_dataloader.dataset)
            print(f"Epoch: {epoch}, Val_Acc: {acc}")


if __name__ == "__main__":
    main()
