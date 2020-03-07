import ast
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from dataset import BengaliDatasetTrain
from model_dispatcher import MODEL_DISPATCHER

DEVICE = os.environ.get('DEVICE', 'cuda')
IMG_HEIGHT = int(os.environ.get('IMG_HEIGHT', 137))
IMG_WEIGHT = int(os.environ.get('IMG_WEIGHT', 236))
EPOCH = int(os.environ.get('EPOCH', 25))

TRAIN_BATCH_SIZE = int(os.environ.get('TRAIN_BATCH_SIZE', 16))
TEST_BATCH_SIZE = int(os.environ.get('TEST_BATCH_SIZE', 8))
PRELOAD_DATASET = os.environ.get('PRELOAD_DATASET', '1') == '1'

BASE_MODEL = os.environ.get('BASE_MODEL', 'resnet18')

DEBUG = os.environ.get('DEBUG', '0') == '1'
if DEBUG:
    n_images = 1000
    TRAIN_FOLDS = (0, 1, 2, 3)
    VAL_FOLDS = (4,)
else:
    TRAIN_FOLDS = ast.literal_eval(os.environ.get('TRAIN_FOLDS'))
    VAL_FOLDS = ast.literal_eval(os.environ.get('TEST_FOLDS'))


def main():
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)
    model.to(DEVICE)

    train_dataset = BengaliDatasetTrain(
        folds=TRAIN_FOLDS,
        transform=T.Compose([
            T.RandomResizedCrop((IMG_HEIGHT, IMG_WEIGHT)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        preload=PRELOAD_DATASET,
        n_images=n_images
    )

    val_dataset = BengaliDatasetTrain(
        folds=VAL_FOLDS,
        transform=T.Compose([
            T.Resize((IMG_HEIGHT, IMG_WEIGHT)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        preload=PRELOAD_DATASET,
        n_images=n_images
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False
    )

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     patience=5,
                                                     factor=0.3, verbose=True)

    model, history = train_model(dataloaders, model, loss_fn, optimizer, scheduler, EPOCH)
    plot_graph(history, EPOCH)


def macro_recall(pred_y, y, n_grapheme=168, n_vowel=11, n_consonant=7):
    pred_y = torch.split(pred_y, [n_grapheme, n_vowel, n_consonant], dim=1)
    pred_labels = [torch.argmax(py, dim=1).cpu().numpy() for py in pred_y]

    y = y.cpu().numpy()

    recall_grapheme = recall_score(pred_labels[0], y[:, 0], average='macro')
    recall_vowel = recall_score(pred_labels[1], y[:, 1], average='macro')
    recall_consonant = recall_score(pred_labels[2], y[:, 2], average='macro')
    final_score = np.average([recall_grapheme, recall_vowel, recall_consonant], weights=[2, 1, 1])

    print(f'recall:grapheme {recall_grapheme}, vowel {recall_vowel}, consonant {recall_consonant}')
    return final_score


def loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    l1 = F.cross_entropy(o1, t1)
    l2 = F.cross_entropy(o2, t2)
    l3 = F.cross_entropy(o3, t3)
    return (l1 + l2 + l3) / 3


def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_macro_recall = 0.0

    history = {
        'train_recall': [],
        'train_loss': [],
        'val_recall': [],
        'val_loss': [],
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            final_outputs = []
            final_targets = []

            # Iterate over data.
            for i, d in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):
                image = d['image'].to(DEVICE)
                grapheme_root = d['grapheme_root'].to(DEVICE)
                vowel_diacritic = d['vowel_diacritic'].to(DEVICE)
                consonant_diacritic = d['consonant_diacritic'].to(DEVICE)

                targets = (grapheme_root, vowel_diacritic, consonant_diacritic)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(image)
                    loss = criterion(outputs, targets)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                final_outputs.append(torch.cat(outputs, dim=1))
                final_targets.append(torch.stack(targets, dim=1))

            final_outputs = torch.cat(final_outputs)
            final_targets = torch.cat(final_targets)
            macro_recall_score = macro_recall(final_outputs, final_targets)
            epoch_loss = running_loss / len(dataloaders[phase])

            # Reduce learning rate when a validation loss has stopped improving.
            if phase == 'val':
                scheduler.step(epoch_loss)

            print(f'{phase} Loss: {epoch_loss:.4f} Macro recall: {macro_recall_score:.4f}')

            history[f'{phase}_recall'].append(macro_recall_score)
            history[f'{phase}_loss'].append(epoch_loss)

            # deep copy the model
            if phase == 'val' and macro_recall_score > best_macro_recall:
                best_macro_recall = macro_recall_score
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f'Best val Macro recall: {best_macro_recall:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


def plot_graph(history, epoch):
    val_loss, val_recall = history['val_loss'], history['val_recall']
    train_loss, train_recall = history['train_loss'], history['train_recall']
    epochsx = np.arange(epoch)

    plt.subplot(3, 1, 1)
    plt.title('Loss on train and validation sets')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochsx, val_loss, '-^', label='val loss')
    plt.plot(epochsx, train_loss, '-o', label='train loss')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.title('Accuracy on train and validation sets')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochsx, val_recall, '-^', label='val recall')
    plt.plot(epochsx, train_recall, '-o', label='train recall')
    plt.legend()


if __name__ == '__main__':
    main()
