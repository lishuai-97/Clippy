import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

total_data, batch_size, input_size, hidden_size = 1000, 16, 1, 32
a = 2

# 初始化权重和偏置
weight1 = torch.randn(hidden_size, input_size) + a
bias1 = torch.randn(hidden_size)
weight2 = torch.randn(hidden_size, hidden_size)
bias2 = torch.randn(hidden_size)
weight3 = torch.randn(1, hidden_size)
bias3 = torch.randn(1)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        # 在构造函数中直接赋值权重和偏置
        with torch.no_grad():
            self.linear1.weight = nn.Parameter(weight1)
            self.linear1.bias = nn.Parameter(bias1)
            self.linear2.weight = nn.Parameter(weight2)
            self.linear2.bias = nn.Parameter(bias2)
            self.linear3.weight = nn.Parameter(weight3)
            self.linear3.bias = nn.Parameter(bias3)

    def forward(self, inputs):
        x = F.relu(self.linear1(inputs))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def train(iters=100, optimizer='adamw', lr=0.01):

    x_data = np.random.randn(total_data, input_size).astype(np.float32)
    y_data = x_data + 3  # y 和 x 是线性关系

    model = Net(input_size, hidden_size)
    model = model.cuda()

    loss_fn = nn.MSELoss(reduction='mean')


    if optimizer == 'AdamW':
        print(f'>>>> Will use AdamW <<<<')
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer == 'ClippyAdamW-v1':
        print('>>>> Will use ClippyAdamW-v1 <<<<')
        from clippy_adamw import ClippyAdamW
        optimizer = ClippyAdamW(model.parameters(), lr=lr)
    elif optimizer == 'ClippyAdamW-v2':
        print('>>>> Will use ClippyAdamW-v2 <<<<')
        from clippy_adamw import ClippyAdamWv2
        optimizer = ClippyAdamWv2(model.parameters(), lr=lr)
    elif optimizer == 'StableAdamWunfused':
        print('>>>> Will use StableAdamWUnfused <<<<')
        from stableadamwunfused import StableAdamWUnfused
        optimizer = StableAdamWUnfused(model.parameters(), lr=lr)
    elif optimizer == 'ClippyAdagrad':
        print('>>>> Will use ClipAdaGrad <<<<')
        from clippy_adagrad_pytorch import ClippyAdagrad
        optimizer = ClippyAdagrad(model.parameters(), lr=lr)
    elif optimizer == 'ClippyAdagrad-v2':
        print('>>>> Will use ClippyAdaGrad-v2 <<<<')
        from clippyadagrad import ClippyAdagrad
        optimizer = ClippyAdagrad(model.parameters(), lr=lr)
    else:
        raise ValueError(f'Invalid optimizer, the optimizer {optimizer} is not supported.')

    loss_list = []  # List to store the loss values

    for t in range(iters):
        idx = np.random.choice(total_data, batch_size, replace=False)
        x = torch.tensor(x_data[idx,:]).cuda()
        label = torch.tensor(y_data[idx,:]).cuda()
        
        pred = model(x)
        loss = loss_fn(pred, label)

        optimizer.zero_grad()
        loss.backward()
        
        print("step: ", t, "    loss: ", loss.item())
        # print("grad: ", model.linear1.weight.grad)
        optimizer.step()
        
        loss_list.append(loss.item())  # Append the loss value to the list

    return loss_list


if __name__ == "__main__":

    # # set random seed
    # torch.manual_seed(0)
    # np.random.seed(0)

    import matplotlib.pyplot as plt

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    adamw_loss = train(iters=args.iteration, optimizer='AdamW')

    clippyadamw_v1_loss = train(iters=args.iteration, optimizer='ClippyAdamW-v1', lr=args.lr)

    clippyadamw_v2_loss = train(iters=args.iteration, optimizer='ClippyAdamW-v2', lr=args.lr)

    clippyadagrad_loss = train(iters=args.iteration, optimizer='ClippyAdagrad', lr=args.lr)

    clippyadagrad_v2_loss = train(iters=args.iteration, optimizer='ClippyAdagrad-v2', lr=args.lr)

    stableadamw_loss = train(iters=args.iteration, optimizer='StableAdamWunfused', lr=args.lr)

    plt.figure()
    plt.plot(range(args.iteration), adamw_loss, label='AdamW')
    # plt.plot(range(args.iteration), clippyadamw_v1_loss, label='ClippyAdamW-v1')
    # plt.plot(range(args.iteration), clippyadamw_v2_loss, label='ClippyAdamW-v2')
    # plt.plot(range(args.iteration), clippyadagrad_loss, label='ClippyAdaGrad')
    # plt.plot(range(args.iteration), clippyadagrad_v2_loss, label='ClippyAdaGrad-v2')
    plt.plot(range(args.iteration), stableadamw_loss, label='StableAdamWUnfused')
    plt.legend()
    # title
    plt.title(f'Loss of {args.iteration} iterations')
    plt.savefig(f'./loss_{args.iteration}_iters.png', dpi=1080)