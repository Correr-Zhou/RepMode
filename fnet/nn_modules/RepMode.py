
import torch
from torch.nn import functional as F
import torch.utils.checkpoint as cp
import math


class Net(torch.nn.Module):
    def __init__(
        self,
        opts,
        mult_chan=32,
        in_channels=1,
        out_channels=1,
    ):
        super().__init__()
        self.opts = opts
        self.mult_chan = mult_chan
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_tasks = len(self.opts.adopted_datasets)
        self.num_experts = 5
        self.gpu_ids = [self.opts.gpu_ids] if isinstance(self.opts.gpu_ids, int) else self.opts.gpu_ids
        self.device = torch.device('cuda', self.gpu_ids[0]) if self.gpu_ids[0] >= 0 else torch.device('cpu')

        # encoder
        self.encoder_block1 = MoDEEncoderBlock(self.num_experts, self.num_tasks, self.in_channels, self.in_channels * self.mult_chan)
        self.encoder_block2 = MoDEEncoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan, self.in_channels * self.mult_chan * 2)
        self.encoder_block3 = MoDEEncoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 2, self.in_channels * self.mult_chan * 4)
        self.encoder_block4 = MoDEEncoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 4, self.in_channels * self.mult_chan * 8)

        # bottle
        self.bottle_block = MoDESubNet2Conv(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 8, self.in_channels * self.mult_chan * 16)

        # decoder
        self.decoder_block4 = MoDEDecoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 16, self.in_channels * self.mult_chan * 8)
        self.decoder_block3 = MoDEDecoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 8, self.in_channels * self.mult_chan * 4)
        self.decoder_block2 = MoDEDecoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 4, self.in_channels * self.mult_chan * 2)
        self.decoder_block1 = MoDEDecoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 2, self.in_channels * self.mult_chan)

        # conv out
        self.conv_out = MoDEConv(self.num_experts, self.num_tasks, self.mult_chan, self.out_channels, kernel_size=5, padding='same', conv_type='final')

    def one_hot_task_embedding(self, task_id):
        N = task_id.shape[0]
        task_embedding = torch.zeros((N, self.num_tasks))
        for i in range(N):
            task_embedding[i, task_id[i]] = 1
        return task_embedding.to(self.device)

    def forward(self, x, t):
        # task embedding
        task_emb = self.one_hot_task_embedding(t)

        # encoding
        x, x_skip1 = self.encoder_block1(x, task_emb)
        x, x_skip2 = self.encoder_block2(x, task_emb)
        x, x_skip3 = self.encoder_block3(x, task_emb)
        x, x_skip4 = self.encoder_block4(x, task_emb)

        # bottle
        x = self.bottle_block(x, task_emb)

        # decoding
        x = self.decoder_block4(x, x_skip4, task_emb)
        x = self.decoder_block3(x, x_skip3, task_emb)
        x = self.decoder_block2(x, x_skip2, task_emb)
        x = self.decoder_block1(x, x_skip1, task_emb)
        outputs = self.conv_out(x, task_emb)

        return outputs


class MoDEEncoderBlock(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, in_chan, out_chan):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.conv_more = MoDESubNet2Conv(num_experts, num_tasks, in_chan, out_chan)
        self.conv_down = torch.nn.Sequential(
            torch.nn.Conv3d(out_chan, out_chan, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm3d(out_chan),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, t):
        x_skip = self.conv_more(x, t)
        x = self.conv_down(x_skip)
        return x, x_skip


class MoDEDecoderBlock(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, in_chan, out_chan):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.convt = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_chan, out_chan, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm3d(out_chan),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_less = MoDESubNet2Conv(num_experts, num_tasks, in_chan, out_chan)

    def forward(self, x, x_skip, t):
        x = self.convt(x)
        x_cat = torch.cat((x_skip, x), 1)  # concatenate
        x_cat = self.conv_less(x_cat, t)
        return x_cat


class MoDESubNet2Conv(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, n_in, n_out):
        super().__init__()
        self.conv1 = MoDEConv(num_experts, num_tasks, n_in, n_out, kernel_size=5, padding='same')
        self.conv2 = MoDEConv(num_experts, num_tasks, n_out, n_out, kernel_size=5, padding='same')

    def forward(self, x, t):
        x = self.conv1(x, t)
        x = self.conv2(x, t)
        return x


class MoDEConv(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, in_chan, out_chan, kernel_size=5, stride=1, padding='same', conv_type='normal'):
        super().__init__()

        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.conv_type = conv_type
        self.stride = stride
        self.padding = padding

        self.expert_conv5x5_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 5)
        self.expert_conv3x3_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 3)
        self.expert_conv1x1_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 1)
        self.register_buffer('expert_avg3x3_pool', self.gen_avgpool_kernel(3))
        self.expert_avg3x3_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 1)
        self.register_buffer('expert_avg5x5_pool', self.gen_avgpool_kernel(5))
        self.expert_avg5x5_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 1)

        assert self.conv_type in ['normal', 'final']
        if self.conv_type == 'normal':
            self.subsequent_layer = torch.nn.Sequential(
                torch.nn.BatchNorm3d(out_chan),
                torch.nn.ReLU(inplace=True),
            )
        else:
            self.subsequent_layer = torch.nn.Identity()

        self.gate = torch.nn.Linear(num_tasks, num_experts * self.out_chan, bias=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def gen_conv_kernel(self, Co, Ci, K):
        weight = torch.nn.Parameter(torch.empty(Co, Ci, K, K, K))
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return weight

    def gen_avgpool_kernel(self, K):
        weight = torch.ones(K, K, K).mul(1.0 / K ** 3)
        return weight

    def trans_kernel(self, kernel, target_size):
        Dp = (target_size - kernel.shape[2]) // 2
        Hp = (target_size - kernel.shape[3]) // 2
        Wp = (target_size - kernel.shape[4]) // 2
        return F.pad(kernel, [Wp, Wp, Hp, Hp, Dp, Dp])

    def routing(self, g, N):

        expert_conv5x5 = self.expert_conv5x5_conv
        expert_conv3x3 = self.trans_kernel(self.expert_conv3x3_conv, self.kernel_size)
        expert_conv1x1 = self.trans_kernel(self.expert_conv1x1_conv, self.kernel_size)
        expert_avg3x3 = self.trans_kernel(
            torch.einsum('oidhw,dhw->oidhw', self.expert_avg3x3_conv, self.expert_avg3x3_pool),
            self.kernel_size,
        )
        expert_avg5x5 = torch.einsum('oidhw,dhw->oidhw', self.expert_avg5x5_conv, self.expert_avg5x5_pool)

        weights = list()
        for n in range(N):
            weight_nth_sample = torch.einsum('oidhw,o->oidhw', expert_conv5x5, g[n, 0, :]) + \
                                torch.einsum('oidhw,o->oidhw', expert_conv3x3, g[n, 1, :]) + \
                                torch.einsum('oidhw,o->oidhw', expert_conv1x1, g[n, 2, :]) + \
                                torch.einsum('oidhw,o->oidhw', expert_avg3x3, g[n, 3, :]) + \
                                torch.einsum('oidhw,o->oidhw', expert_avg5x5, g[n, 4, :])
            weights.append(weight_nth_sample)
        weights = torch.stack(weights)

        return weights

    def forward(self, x, t):

        N = x.shape[0]

        g = self.gate(t)
        g = g.view((N, self.num_experts, self.out_chan))
        g = self.softmax(g)

        w = self.routing(g, N)

        if self.training:
            y = list()
            for i in range(N):
                y.append(F.conv3d(x[i].unsqueeze(0), w[i], bias=None, stride=1, padding='same'))
            y = torch.cat(y, dim=0)
        else:
            y = F.conv3d(x, w[0], bias=None, stride=1, padding='same')

        y = self.subsequent_layer(y)

        return y
