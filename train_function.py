import os
import torch
import torch.optim as optim
import time
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from scheduler import CosineAnnealingWarmUpRestarts


def train_fn(args, model, device, train_dataset, valid_dataset, criterion, fold, writer, LOGGER):
    scaler = GradScaler()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9, eps=1e-8, weight_decay=args.weight_decay, momentum=args.momentum)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-7)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=1, eta_max=args.lr, T_up=10, gamma=0.7)

    best_loss = np.inf
    best_accuracy = 0

    start_time = time.time()

    for epoch in range(args.epochs*fold + 1, args.epochs*fold + args.epochs + 1):
        model.train()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                                   pin_memory=True, num_workers=args.num_workers)

        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.test_batch_size, shuffle=False,
                                                   pin_memory=True, num_workers=args.num_workers)

        train_loss, train_accuracy, train_lr = train_epoch(epoch, args, model, device, train_loader, criterion,
                                                           optimizer, scheduler, scaler, LOGGER)

        writer.add_scalars('Loss', {'train': train_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_accuracy}, epoch)
        writer.add_scalars('LR', {'train': train_lr}, epoch)

        if epoch % args.val_per_epochs == 0:
            # print("=====Validation=====")
            LOGGER.info(f'\n========== Validation ==========')
            valid_loss, valid_accuracy = test_epoch(args, model, device, valid_loader, criterion, LOGGER)
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, f'{args.model_name}_fold{fold}_best_accuracy.pt'))
                LOGGER.info(f'Save the best acc model, loss = {valid_loss:2f}, acc = {best_accuracy:2f}')

            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, f'{args.model_name}_fold{fold}_best_loss.pt'))
                LOGGER.info(f'Save the best loss model, loss = {best_loss:2f}, acc = {valid_accuracy:2f}')

            writer.add_scalars('Loss', {'val': valid_loss}, epoch)
            writer.add_scalars('Accuracy', {'val': valid_accuracy}, epoch)

            # model.train()

    end_time = time.time()
    # LOGGER.info("\n Total Training Time : {}".format(end_time-start_time))


def test_fn(args, model, device, dataset, criterion, LOGGER=None):
    # start_time = time.time()
    LOGGER.info("\n=====Test=====")
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False,
                                              pin_memory=False, num_workers=args.num_workers)
    loss, accuracy = test_epoch(args, model, device, test_loader, criterion, LOGGER)
    # end_time = time.time()
    # LOGGER.info(f'Evaluation Time : {end_time - start_time}')
    return loss, accuracy


def train_epoch(epoch, args, model, device, train_loader, criterion, optimizer, scheduler, scaler, LOGGER=None):
    train_loss = []
    correct = 0

    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        # target = torch.zeros(len(target), target.max() + 1).scatter_(1, target.unsqueeze(1), 1.0) # one-hot encoding
        target = target.to(device)

        optimizer.zero_grad()
        with autocast():
            output = model(data.to(device))
            loss = criterion(output, target)

        # output = model(data.to(device))
        # loss = criterion(output, target.unsqueeze(1).float().to(device))

        # get loss
        train_loss.append(loss.item())

        # get preds (for accuracy)
        pred = output.max(1)[1]
        correct += pred.eq(target).sum().item()

        # loss.backward()
        scaler.scale(loss).backward()

        # # to apply scheduler
        scaler.unscale_(optimizer)
        scale_before = scaler.get_scale()

        # optimizer.step()
        scaler.step(optimizer)

        # scaler update
        scaler.update()

        # scheduler.step()
        scale_after = scaler.get_scale()
        if scale_before <= scale_after:
            scheduler.step()

        if batch_idx % args.log_interval == 0:
            LOGGER.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                        f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.8f}')

            if args.dry_run:
                break

    mean_loss = sum(train_loss) / (len(train_loss) + 1e-7)
    accuracy = 100. * correct / len(train_loader.dataset)
    LR = optimizer.param_groups[0]['lr']

    end_time = time.time()
    LOGGER.info(f'Train Mean Loss : {mean_loss:.8f},\tTrain Accuracy: {accuracy:.2f},'
                f'\tTime : {end_time - start_time:.2f} sec\t LR : {LR:.8f}')

    return mean_loss, accuracy, LR


def test_epoch(args, model, device, data_loader, criterion, LOGGER=None):
    model.eval()
    test_loss = 0
    correct = 0

    data_len = len(data_loader.dataset)

    with torch.no_grad():
        for step, (data, target) in enumerate(data_loader):
            # target = torch.zeros(len(target), target.max() + 1).scatter_(1, target.unsqueeze(1), 1.)  # one-hot encoding
            target = target.to(device)
            with torch.no_grad():
                output = model(data.to(device))
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1)[1]  # get the index of the max log-probability
            # pred = torch.round(output)
            correct += pred.eq(target).sum().item()

    try:
        accuracy = 100. * correct / data_len
    except:
        accuracy = 100. * correct / (data_len + 1e-8)

    try:
        test_loss /= (step / args.num_workers)
    except:
        test_loss /= (19 / args.num_workers)
    LOGGER.info(f'Average loss: {test_loss:.8f}, Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.2f}%)')

    return test_loss, accuracy
