import torch
import torch.nn as nn
from facenet_pytorch import InceptionResNetV1


class FaceResNet(nn.Module):
    def __init__(self, args):
        super(FaceResNet).__init__()
        self.resnet = InceptionResNetV1(pretrained="vggface2")
        self.linear = nn.Linear(512, args.emb)
        self.net = nn.Sequential(self.resnet, self.linear)

    def forward(self, x):
        return self.net(x)


class VisualLstm(nn.Module):
    def __init__(self, args):
        super(VisualLstm).__init__()
        self.lstm = nn.LSTM(
            input_size=args.emb,
            hidden_size=args.emb,
            dropout=0.2,
            bidirectional=False,
            num_layers=args.visual_lstm_layer,
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return output[-1]


class LangLstm(nn.Module):
    def __init__(self, args):
        super(LangLstm).__init__()
        self.lstm = nn.LSTM(
            input_size=args.emb,
            hidden_size=args.emb,
            dropout=0.2,
            bidirectional=False,
            num_layers=args.text_lstm_layer,
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return output[-1]


class MultiModel(nn.Module):
    def __init__(self, args):
        super(MultiModel).__init__()
        self.fc1 = nn.Linear(args.emb * 2, args.emb)
        self.fc2 = nn.Linear(args.emb, 7)
        self.relu = nn.ReLU(inplace=True)
        self.net = nn.Sequential(self.fc1, self.relu, self.fc2)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.net(x)
