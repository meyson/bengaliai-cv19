import ast
import copy
import os

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BengaliDatasetTrain
from model_dispatcher import MODEL_DISPATCHER

DEVICE = os.environ.get('DEVICE', 'cuda')
IMG_HEIGHT = int(os.environ.get('IMG_HEIGHT', 137))
IMG_WEIGHT = int(os.environ.get('IMG_WEIGHT', 236))
EPOCH = int(os.environ.get('EPOCH', 25))

TRAIN_BATCH_SIZE = int(os.environ.get('TRAIN_BATCH_SIZE', 16))
TEST_BATCH_SIZE = int(os.environ.get('TEST_BATCH_SIZE', 8))
PRELOAD_DATASET = os.environ.get('PRELOAD_DATASET', '0') == '1'

BASE_MODEL = os.environ.get('BASE_MODEL', 'squeezenet')

TRAINING_FOLDS = ast.literal_eval(os.environ.get('TRAINING_FOLDS', '(0, 1, 2, 3)'))
VALIDATION_FOLDS = ast.literal_eval(os.environ.get('VALIDATION_FOLDS', '(4, )'))

RGB = os.environ.get('RGB', '0') == '1'


def main():
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True, RGB=RGB)
    model.to(DEVICE)

    ImageNetStat = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }

    BengaliAIStat = {
        'mean': [0.06922848809290576],
        'std': [0.20515700083327537]
    }

    stat = ImageNetStat if RGB else BengaliAIStat

    train_dataset = BengaliDatasetTrain(
        folds=TRAINING_FOLDS,
        aug=A.Compose([
            A.Resize(IMG_HEIGHT, IMG_HEIGHT, always_apply=True),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.2,
                rotate_limit=5,
                p=0.9
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.4, p=1),
            A.Normalize(**stat)
        ]),
        preload=PRELOAD_DATASET,
        RGB=RGB
    )

    val_dataset = BengaliDatasetTrain(
        folds=VALIDATION_FOLDS,
        aug=A.Compose([
            A.Resize(IMG_HEIGHT, IMG_WEIGHT, always_apply=True),
            A.Normalize(**stat)
        ]),
        preload=PRELOAD_DATASET,
        RGB=RGB
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print('N of parameters:', count_parameters(model))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     patience=5,
                                                     factor=0.3, verbose=True)

    print(model)
    model, history = train_model(train_loader, val_loader, model, loss_fn, optimizer, scheduler, num_epochs=EPOCH)
    plot_graph(history, EPOCH)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def macro_recall(pred_y, y, n_grapheme=168, n_vowel=11, n_consonant=7, verbose=False):
    pred_y = torch.split(pred_y, [n_grapheme, n_vowel, n_consonant], dim=1)
    pred_labels = [torch.argmax(py, dim=1).cpu().numpy() for py in pred_y]

    y = y.cpu().numpy()

    recall_grapheme = recall_score(pred_labels[0], y[:, 0], average='macro')
    recall_vowel = recall_score(pred_labels[1], y[:, 1], average='macro')
    recall_consonant = recall_score(pred_labels[2], y[:, 2], average='macro')
    final_score = np.average([recall_grapheme, recall_vowel, recall_consonant], weights=[2, 1, 1])
    if verbose:
        print(f'recall: grapheme {recall_grapheme}, vowel {recall_vowel}, consonant {recall_consonant}')
    return final_score


def loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    l1 = F.cross_entropy(o1, t1)
    l2 = F.cross_entropy(o2, t2)
    l3 = F.cross_entropy(o3, t3)
    return (l1 + l2 + l3) / 3


def check_accuracy(loader, model, criterion):
    model.eval()

    running_loss = 0.0
    final_outputs = []
    final_targets = []

    with torch.no_grad():
        for i, d in tqdm(enumerate(loader), total=len(loader)):
            image = d['image'].to(DEVICE)
            grapheme_root = d['grapheme_root'].to(DEVICE)
            vowel_diacritic = d['vowel_diacritic'].to(DEVICE)
            consonant_diacritic = d['consonant_diacritic'].to(DEVICE)
            targets = (grapheme_root, vowel_diacritic, consonant_diacritic)

            outputs = model(image)
            loss = criterion(outputs, targets)
            # statistics
            running_loss += loss.item()
            final_outputs.append(torch.cat(outputs, dim=1))
            final_targets.append(torch.stack(targets, dim=1))

        final_outputs = torch.cat(final_outputs)
        final_targets = torch.cat(final_targets)
        val_macro_recall = macro_recall(final_outputs, final_targets)
        val_loss = running_loss / len(loader)
        return val_loss, val_macro_recall


def train_model(train_loader, val_loader, model, criterion,
                optimizer, scheduler, history=None, num_epochs=25):
    state_path = f'models/{BASE_MODEL}_train_folds_{train_loader.dataset.folds}{"_rgb" if RGB else ""}.h5'
    if os.path.isfile(state_path):
        print(f'Checking accuracy of {state_path}')
        best_model_wts = model.load_state_dict(torch.load(state_path))
        _, best_macro_recall = check_accuracy(val_loader, model, criterion)
        print(f'best_macro_recall {best_macro_recall}')
    else:
        best_model_wts = copy.deepcopy(model.state_dict())
        best_macro_recall = 0.0

    if history is None:
        history = {'train_recall': [], 'train_loss': [], 'val_recall': [], 'val_loss': []}

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        running_loss = 0.0
        final_outputs = []
        final_targets = []

        # Iterate over data.
        for i, d in tqdm(enumerate(train_loader), total=len(train_loader)):
            image = d['image'].to(DEVICE)
            grapheme_root = d['grapheme_root'].to(DEVICE)
            vowel_diacritic = d['vowel_diacritic'].to(DEVICE)
            consonant_diacritic = d['consonant_diacritic'].to(DEVICE)

            targets = (grapheme_root, vowel_diacritic, consonant_diacritic)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(image)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            final_outputs.append(torch.cat(outputs, dim=1))
            final_targets.append(torch.stack(targets, dim=1))

        final_outputs = torch.cat(final_outputs)
        final_targets = torch.cat(final_targets)

        train_macro_recall = macro_recall(final_outputs, final_targets)
        train_loss = running_loss / len(train_loader)
        print(f'Train Loss: {train_loss:.4f} Macro recall: {train_macro_recall:.4f}')

        val_loss, val_macro_recall = check_accuracy(val_loader, model, criterion)
        print(f'Val Loss: {val_loss:.4f} Macro recall: {val_macro_recall:.4f}')

        scheduler.step(val_macro_recall)

        history['train_loss'].append(train_loss)
        history['train_recall'].append(train_macro_recall)
        history['val_loss'].append(val_loss)
        history['val_recall'].append(val_macro_recall)

        # deep copy the model
        if val_macro_recall > best_macro_recall:
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), state_path)
            best_macro_recall = val_macro_recall

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
