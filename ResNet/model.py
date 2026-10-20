import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels: int = 3, labels: int = 5, channels: int = 256):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.labels = labels
        self.channels = channels

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.reshape = nn.Conv2d(in_channels, channels, kernel_size=5, stride=3, padding=2)
        self.dropout = nn.Dropout2d(0.2)

        self.norm2 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.norm3 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.norm4 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.norm5 = nn.BatchNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.norm6 = nn.BatchNorm2d(channels)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.norm7 = nn.BatchNorm2d(channels)
        self.conv6 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.pool6 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.norm8 = nn.BatchNorm2d(channels)
        self.conv7 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.pool7 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.norm9 = nn.BatchNorm2d(channels)
        self.conv8 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.pool8 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.norm10 = nn.BatchNorm2d(channels)
        self.conv9 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.pool9 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.norm11 = nn.BatchNorm2d(channels)
        self.conv10 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.pool10 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.norm12 = nn.BatchNorm2d(channels)
        self.conv11 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.pool11 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.norm13 = nn.BatchNorm2d(channels)
        self.conv12 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.pool12 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.norm14 = nn.BatchNorm2d(channels)
        self.conv13 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Linear(channels, labels)

    def forward(self, x):
        x = self.dropout(F.relu(self.reshape(self.norm1(x))))
        skip = x
        x = F.relu(self.conv1(self.norm2(x))) + skip
        x = self.pool1(x)
        skip = x
        x = F.relu(self.conv2(self.norm3(x))) + skip
        x = self.pool2(x)
        skip = x
        x = F.relu(self.conv3(self.norm4(x))) + skip
        x = self.pool3(x)
        skip = x
        x = F.relu(self.conv4(self.norm5(x))) + skip
        x = self.pool4(x)
        skip = x
        x = F.relu(self.conv5(self.norm6(x))) + skip
        # x = self.pool5(x)
        x = F.relu(self.conv6(self.norm7(x))) + skip
        x = self.pool6(x)
        skip = x
        x = F.relu(self.conv7(self.norm8(x))) + skip
        # x = self.pool7(x)
        skip = x
        x = F.relu(self.conv8(self.norm9(x))) + skip
        # x = self.pool8(x)
        skip = x
        x = F.relu(self.conv9(self.norm10(x))) + skip
        # x = self.pool9(x)
        skip = x
        x = F.relu(self.conv10(self.norm11(x))) + skip
        # x = self.pool10(x)
        skip = x
        x = F.relu(self.conv11(self.norm12(x))) + skip
        # x = self.pool11(x)
        skip = x
        x = F.relu(self.conv12(self.norm13(x))) + skip
        # x = self.pool12(x)
        skip = x
        x = F.relu(self.conv13(self.norm14(x))) + skip
        # x = self.pool4(x)
        return self.out(self.flatten(x).flatten(1))


    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, (nn.Conv1d, nn.Linear)):
    #             torch.nn.init.kaiming_normal_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)
    #     nn.init.xavier_uniform_(self.output.weight)
    #     nn.init.zeros_(self.output.bias)



