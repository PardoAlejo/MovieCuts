import torch
import torch.nn as nn
import torch.nn.functional as F
from video_resnet import r2plus1d_18
from resnet import resnet18


class AudioVisualModel(nn.Module):
    def __init__(self, num_classes, mlp=True, **kwargs):
        super(AudioVisualModel, self).__init__()
        self.audnet = resnet18(num_classes=num_classes, last_fc=False, **kwargs)
        self.r2p1d = r2plus1d_18(num_classes=num_classes, last_fc=False, **kwargs)

        self.mlp = mlp
        if self.mlp:
            #shared
            self.relu = nn.ReLU(inplace=True)
            self.fc_128_audnet = nn.Linear(512, 128)
            self.fc_128_r2p1d = nn.Linear(512, 128)

            # Predictions
            self.fc_final = nn.Linear(128*2, num_classes)
            self.fc_aux_audnet = nn.Linear(128, num_classes)
            self.fc_aux_r2p1d = nn.Linear(128, num_classes)
        else:
            self.fc_final = nn.Linear(512*2, num_classes)
            self.fc_aux_audnet = nn.Linear(512, num_classes)
            self.fc_aux_r2p1d = nn.Linear(512, num_classes)

    def forward(self, visual, audio):
        
        audio = self.audnet(audio)
        visual = self.r2p1d(visual)
        # audio_visual_feats = torch.cat((audio, visual), 1)

        if self.mlp:
            audio = self.fc_128_audnet(audio)
            audio = self.relu(audio)
            visual = self.fc_128_r2p1d(visual)
            visual = self.relu(visual)

            out_audio = self.fc_aux_audnet(audio)
            out_visual = self.fc_aux_r2p1d(visual)

        else:
            out_audio = self.fc_aux_audnet(audio)
            out_visual = self.fc_aux_r2p1d(visual)

        audio_visual = torch.cat((audio, visual), 1)
        preds = self.fc_final(audio_visual)

        return preds, out_visual, out_audio