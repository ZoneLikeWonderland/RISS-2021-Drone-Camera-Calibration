import torch
import torch.nn as nn
import torchvision.models as models
from unet import UNet
from dataset import *
from torch.utils.tensorboard import SummaryWriter
import tqdm
from torch import optim
import time
import os


def test_net(net):
    net.eval()
    # criterion = nn.MSELoss()

    # losses = []
    # count = 0
    # for gt, bg, pt in tqdm.tqdm(loader, disable=True):
    bgs = []
    for path in glob.glob(r"E:\RISS\test_images\*.jpg") + glob.glob(r"E:\RISS\test_images\*.png"):
        # gt = gt.cuda()
        # bg = bg.cuda()
        # pt = pt.cuda()
        img = cv2.imread(path).astype(np.float32) / 255
        img = cv2.resize(img, (WIDTH, HEIGHT), cv2.INTER_AREA)
        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
        bg = img.cuda()
        bgs.append(bg)

    bg = torch.cat(bgs, dim=0)

    with torch.no_grad():
        fg = net(bg)

    show = np.concatenate((
        bg.detach().cpu().numpy(),
        # np.repeat(gt.detach().cpu().numpy(), 3, 1),
        np.repeat(fg[:, :1].detach().cpu().numpy(), 3, 1),
        # np.repeat(pt.detach().cpu().numpy(), 3, 1),
        np.repeat(fg[:, 1:].detach().cpu().numpy(), 3, 1),
    ), axis=2).clip(0, 1)
    writer.add_images('show/test', show[:, ::-1], global_step)

    net.train()
    # return np.mean(losses)


def eval_net(net, loader):
    net.eval()
    criterion = nn.MSELoss()

    losses = []
    count = 0
    for gt, bg, pt in tqdm.tqdm(loader, disable=True):
        gt = gt.cuda()
        bg = bg.cuda()
        pt = pt.cuda()
        with torch.no_grad():
            fg = net(bg)
        loss = criterion(fg[:, :1], gt) + criterion(fg[:, 1:], pt)
        losses.append(loss.item())
        count += 1
        if count > 10:
            break

    show = np.concatenate((
        bg.detach().cpu().numpy(),
        np.repeat(gt.detach().cpu().numpy(), 3, 1),
        np.repeat(fg[:, :1].detach().cpu().numpy(), 3, 1),
        np.repeat(pt.detach().cpu().numpy(), 3, 1),
        np.repeat(fg[:, 1:].detach().cpu().numpy(), 3, 1),
    ), axis=2).clip(0, 1)
    writer.add_images('show/eval', show[:, ::-1], global_step)

    net.train()
    return np.mean(losses)


def train(net, lr=0.001, epochs=1000, batch_size=8, comment=""):
    global writer, global_step
    net.cuda()
    net.train()

    card_set = CardSet()
    train_loader = torch.utils.data.DataLoader(card_set, num_workers=8, shuffle=True, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(card_set, num_workers=0, shuffle=True, batch_size=batch_size)
    n_train = len(card_set)

    writer = SummaryWriter(comment=comment)
    global_step = 0

    criterion = nn.MSELoss()
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)

    last_show_time = 0
    # last_test_time = time.time()
    last_test_time = 0
    best_test_error = np.inf
    for epoch in range(epochs):
        for gt, bg, pt in tqdm.tqdm(train_loader, disable=True):
            gt = gt.cuda()
            bg = bg.cuda()
            pt = pt.cuda()

            fg = net(bg)

            loss = criterion(fg[:, :1], gt) + criterion(fg[:, 1:], pt)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1

            if time.time() - last_show_time > 5:
                last_show_time = time.time()

                show = np.concatenate((
                    bg.detach().cpu().numpy(),
                    np.repeat(gt.detach().cpu().numpy(), 3, 1),
                    np.repeat(fg[:, :1].detach().cpu().numpy(), 3, 1),
                    np.repeat(pt.detach().cpu().numpy(), 3, 1),
                    np.repeat(fg[:, 1:].detach().cpu().numpy(), 3, 1),
                ), axis=2).clip(0, 1)
                writer.add_images('show/train', show[:, ::-1], global_step)

            if time.time() - last_test_time > 3 * 60:
                last_test_time = time.time()
                print("test")
                test_net(net)
                val_error = eval_net(net, test_loader)
                scheduler.step(val_error)
                writer.add_scalar('Loss/test', val_error, global_step)

                if val_error < best_test_error:
                    best_test_error = val_error
                    torch.save(net, os.path.join(writer.file_writer.get_logdir(), "best_test_error.pth"))


if __name__ == "__main__":
    net = UNet(3, 2, base=20)
    net = torch.load(r"E:\RISS\runs\Jul06_13-06-00_DESKTOP-HN2581Fwithtest20\best_test_error.pth")
    comment = "withtest20"
    train(net, comment=comment)
