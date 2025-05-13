from typing import Dict

import torch
from tqdm import tqdm

from utils.loss import crossentropy, soft_crossentropy
from utils.metrics import accuracy

def training_step(
    model, dataloader, optimizer, scheduler, scaler,
    history : Dict, accumulation_steps : int = 4, device : str = 'cuda'
    ):
    model.train()
    pbar = tqdm(dataloader, f"Train loss: {history['train_loss'][-1]:.3f}, acc: {history['train_acc'][-1]:.3f}")
    total_loss = 0
    cur_acc = 0
    total_acc = 0
    out_of_mem = 0

    for batch_idx, (images, labels) in enumerate(pbar):
        try:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast('cuda'):
                prediction = model(images)
                loss_value = soft_crossentropy(prediction, labels)
            total_loss += loss_value.item()

            scaler.scale(loss_value/accumulation_steps).backward()

            if (batch_idx+1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if batch_idx % 10 == 0:
                pbar.set_description(f"Train loss: {loss_value.item():.3f}, acc: {cur_acc:.3f}")
        except RuntimeError as e:
            print(e)
            if 'out of memory' in str(e):
                out_of_mem += 1
                torch.cuda.empty_cache()
            else:
                raise e
        finally:
            scheduler.step()
    else:
        if (batch_idx+1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad()
    mean_loss = total_loss/(batch_idx + 1)
    history['train_loss'].append(mean_loss)
    mean_acc = total_acc/(batch_idx + 1)
    history['train_acc'].append(mean_acc)
    if out_of_mem:
        print(f"Memory overflow occurred in {out_of_mem}/{batch_idx+1} batches!")
    return mean_loss, mean_acc

@torch.no_grad()
def validation_step(model, dataloader, history : Dict, device : str = 'cuda'):
    model.eval()
    pbar = tqdm(dataloader, f"Validation loss: {history['val_loss'][-1]:.3f}")
    total_loss = 0
    total_acc = 0
    out_of_mem = 0
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        try:
            prediction = model(images)
            loss_value = crossentropy(prediction, labels)
            total_loss += loss_value.item()
            cur_acc = accuracy(prediction, labels)
            total_acc += cur_acc
            if batch_idx % 10 == 0:
                pbar.set_description(f"Validation loss: {loss_value.item():.3f}, acc: {cur_acc:.3f}")
        except RuntimeError as e:

            if 'out of memory' in str(e):
                out_of_mem += 1
                torch.cuda.empty_cache()
            else:
                raise e


    mean_loss = total_loss/(batch_idx + 1)
    history['val_loss'].append(mean_loss)
    mean_acc = total_acc/(batch_idx + 1)
    history['val_acc'].append(mean_acc)
    if out_of_mem:
        print(f"Memory overflow occurred in {out_of_mem}/{batch_idx+1} batches!")
    return mean_loss, mean_acc
