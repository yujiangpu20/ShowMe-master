import torch
import torch.nn as nn
import torch.nn.functional as F


class HedEdge(nn.Module):
    def __init__(self):
        super().__init__()

        self.netVggOne = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.netVggTwo = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.netVggThr = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.netVggFou = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.netVggFiv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.netScoreOne = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * 255.0
        mean = torch.tensor(data=[104.00698793, 116.66876762, 122.67891434], device=x.device).view(1, 3, 1, 1)
        x = x - mean

        x1 = self.netVggOne(x)
        x2 = self.netVggTwo(x1)
        x3 = self.netVggThr(x2)
        x4 = self.netVggFou(x3)
        x5 = self.netVggFiv(x4)

        s1 = self.netScoreOne(x1)
        s2 = self.netScoreTwo(x2)
        s3 = self.netScoreThr(x3)
        s4 = self.netScoreFou(x4)
        s5 = self.netScoreFiv(x5)

        s1 = F.interpolate(input=s1, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        s2 = F.interpolate(input=s2, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        s3 = F.interpolate(input=s3, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        s4 = F.interpolate(input=s4, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        s5 = F.interpolate(input=s5, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

        return self.netCombine(torch.cat([s1, s2, s3, s4, s5], 1))