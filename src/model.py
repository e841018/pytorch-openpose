from collections import OrderedDict
import torch, torch.nn as nn

def make_layers(name_spec_list, no_relu_layers):
    layers = []
    for name, spec in name_spec_list:
        if 'pool' in name:
            layers.append((name, nn.MaxPool2d(
                kernel_size=spec[0],
                stride=spec[1],
                padding=spec[2])))
        else:
            layers.append((name, nn.Conv2d(
                in_channels=spec[0],
                out_channels=spec[1],
                kernel_size=spec[2],
                stride=spec[3],
                padding=spec[4])))
            if name not in no_relu_layers:
                layers.append(('relu_'+name, nn.ReLU(inplace=True)))
    return nn.Sequential(OrderedDict(layers))

class bodypose_model(nn.Module):
    def __init__(self):
        super(bodypose_model, self).__init__()

        # these layers have no relu layer
        no_relu_layers = {
            'conv5_5_CPM_L1', 'conv5_5_CPM_L2',
            'Mconv7_stage2_L1', 'Mconv7_stage2_L2',
            'Mconv7_stage3_L1', 'Mconv7_stage3_L2',
            'Mconv7_stage4_L1', 'Mconv7_stage4_L2',
            'Mconv7_stage5_L1', 'Mconv7_stage5_L2',
            'Mconv7_stage6_L1', 'Mconv7_stage6_L2'}

        # Stage 0 (backbone)
        self.stage0 = make_layers([
                ('conv1_1',             [  3,  64, 3, 1, 1]),
                ('conv1_2',             [ 64,  64, 3, 1, 1]),
                ('pool1_stage1',                  [2, 2, 0]),
                ('conv2_1',             [ 64, 128, 3, 1, 1]),
                ('conv2_2',             [128, 128, 3, 1, 1]),
                ('pool2_stage1',                  [2, 2, 0]),
                ('conv3_1',             [128, 256, 3, 1, 1]),
                ('conv3_2',             [256, 256, 3, 1, 1]),
                ('conv3_3',             [256, 256, 3, 1, 1]),
                ('conv3_4',             [256, 256, 3, 1, 1]),
                ('pool3_stage1',                  [2, 2, 0]),
                ('conv4_1',             [256, 512, 3, 1, 1]),
                ('conv4_2',             [512, 512, 3, 1, 1]),
                ('conv4_3_CPM',         [512, 256, 3, 1, 1]),
                ('conv4_4_CPM',         [256, 128, 3, 1, 1])], no_relu_layers)

        # Stage 1
        self.stage1_1 = make_layers([
                ('conv5_1_CPM_L1',      [128, 128, 3, 1, 1]),
                ('conv5_2_CPM_L1',      [128, 128, 3, 1, 1]),
                ('conv5_3_CPM_L1',      [128, 128, 3, 1, 1]),
                ('conv5_4_CPM_L1',      [128, 512, 1, 1, 0]),
                ('conv5_5_CPM_L1',      [512,  38, 1, 1, 0])], no_relu_layers)
        self.stage1_2 = make_layers([
                ('conv5_1_CPM_L2',      [128, 128, 3, 1, 1]),
                ('conv5_2_CPM_L2',      [128, 128, 3, 1, 1]),
                ('conv5_3_CPM_L2',      [128, 128, 3, 1, 1]),
                ('conv5_4_CPM_L2',      [128, 512, 1, 1, 0]),
                ('conv5_5_CPM_L2',      [512,  19, 1, 1, 0])], no_relu_layers)

        # Stages 2 - 6
        for b in range(2, 7):
            self.add_module(f'stage{b}_1', make_layers([
                (f'Mconv1_stage{b}_L1', [185, 128, 7, 1, 3]),
                (f'Mconv2_stage{b}_L1', [128, 128, 7, 1, 3]),
                (f'Mconv3_stage{b}_L1', [128, 128, 7, 1, 3]),
                (f'Mconv4_stage{b}_L1', [128, 128, 7, 1, 3]),
                (f'Mconv5_stage{b}_L1', [128, 128, 7, 1, 3]),
                (f'Mconv6_stage{b}_L1', [128, 128, 1, 1, 0]),
                (f'Mconv7_stage{b}_L1', [128,  38, 1, 1, 0])], no_relu_layers))
            self.add_module(f'stage{b}_2', make_layers([
                (f'Mconv1_stage{b}_L2', [185, 128, 7, 1, 3]),
                (f'Mconv2_stage{b}_L2', [128, 128, 7, 1, 3]),
                (f'Mconv3_stage{b}_L2', [128, 128, 7, 1, 3]),
                (f'Mconv4_stage{b}_L2', [128, 128, 7, 1, 3]),
                (f'Mconv5_stage{b}_L2', [128, 128, 7, 1, 3]),
                (f'Mconv6_stage{b}_L2', [128, 128, 1, 1, 0]),
                (f'Mconv7_stage{b}_L2', [128,  19, 1, 1, 0])], no_relu_layers))

    def forward(self, x):

        out0 = self.stage0(x)

        out1_1 = self.stage1_1(out0)
        out1_2 = self.stage1_2(out0)
        out1 = torch.cat([out1_1, out1_2, out0], 1)

        out2_1 = self.stage2_1(out1)
        out2_2 = self.stage2_2(out1)
        out2 = torch.cat([out2_1, out2_2, out0], 1)

        out3_1 = self.stage3_1(out2)
        out3_2 = self.stage3_2(out2)
        out3 = torch.cat([out3_1, out3_2, out0], 1)

        out4_1 = self.stage4_1(out3)
        out4_2 = self.stage4_2(out3)
        out4 = torch.cat([out4_1, out4_2, out0], 1)

        out5_1 = self.stage5_1(out4)
        out5_2 = self.stage5_2(out4)
        out5 = torch.cat([out5_1, out5_2, out0], 1)

        out6_1 = self.stage6_1(out5)
        out6_2 = self.stage6_2(out5)
        out6 = torch.cat([out6_1, out6_2], 1)

        return out6

# transfer caffe model layer names to new names defined in model.py
def transfer(model_weights, new_names):
    state_dict = {}
    for new_name in new_names:
        old_name = new_name[new_name.index('.')+1:]
        state_dict[new_name] = model_weights[old_name]
    return state_dict
