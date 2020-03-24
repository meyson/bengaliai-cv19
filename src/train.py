import ast
import glob
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
VAL_BATCH_SIZE = int(os.environ.get('VAL_BATCH_SIZE', 16))
PRELOAD_DATASET = os.environ.get('PRELOAD_DATASET', '0') == '1'

BASE_MODEL = os.environ.get('BASE_MODEL', 'squeezenet')
CHECKPOINT = os.environ.get('CHECKPOINT', '')

TRAINING_FOLDS = ast.literal_eval(os.environ.get('TRAINING_FOLDS', '(0, 1, 2, 3)'))
VALIDATION_FOLDS = ast.literal_eval(os.environ.get('VALIDATION_FOLDS', '(4, )'))

USE_RGB = os.environ.get('USE_RGB', '0') == '1'

ImageNetStat = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

BengaliAIStat = {
    'mean': [0.06922848809290576],
    'std': [0.20515700083327537]
}

STAT = ImageNetStat if USE_RGB else BengaliAIStat


def main():
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True, use_rgb=USE_RGB)
    model.to(DEVICE)

    train_dataset = BengaliDatasetTrain(
        folds=TRAINING_FOLDS,
        aug=A.Compose([
            A.Resize(IMG_HEIGHT, IMG_HEIGHT, always_apply=True),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=0,
                p=0.2
            ),
            # A.GridDistortion(num_steps=5, distort_limit=0.5, p=1),
            # A.CoarseDropout(p=0.1),
            # A.ElasticTransform(alpha=0, sigma=50, alpha_affine=13, p=0.1),
            A.GridDistortion(num_steps=5, distort_limit=0.5, p=0.5),
            # A.CoarseDropout(max_holes=5, max_height=15, max_width=15, p=0.1),
            A.Normalize(**STAT),
            A.CoarseDropout(max_holes=2, max_height=36, max_width=56, p=0.5, fill_value=0)
        ]),
        preload=PRELOAD_DATASET,
        data_format='pkl',
        use_rgb=USE_RGB
    )

    val_dataset = BengaliDatasetTrain(
        folds=VALIDATION_FOLDS,
        aug=A.Compose([
            A.Resize(IMG_HEIGHT, IMG_WEIGHT, always_apply=True),
            A.Normalize(**STAT)
        ]),
        preload=PRELOAD_DATASET,
        data_format='pkl',
        use_rgb=USE_RGB
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-7)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='max',
                                                     patience=5,
                                                     factor=0.3, verbose=True)
    if CHECKPOINT:
        print('Loading checkpoint...')
        print(model.load_state_dict(torch.load(CHECKPOINT)))

    # freeze new layers
    # not_frozen = ['model.features.0', 'l0', 'l1', 'l2']
    # for name, param in model.named_parameters():
    #     if any(p in name for p in not_frozen):
    #         continue
    #     param.requires_grad = False

    print('N of parameters:', count_parameters(model))

    model, history = train_model(train_loader, val_loader, model, loss_fn,
                                 optimizer, scheduler, num_epochs=EPOCH)
    check_pretrained()
    plot_graph(history, EPOCH)


def get_state_path(train_folds):
    return f'pretrained_models/{BASE_MODEL}_train_folds_{train_folds}{"_rgb" if USE_RGB else ""}.h5'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_pretrained(train_folds=None, val_folds=None):
    if not train_folds or not val_folds:
        train_folds = [
            (0, 1, 2, 3),
            (0, 1, 2, 4),
            (0, 1, 3, 4),
            (0, 2, 3, 4),
            (1, 2, 3, 4)
        ]
        val_folds = [(4,), (3,), (2,), (1,), (0,)]

    for train_fold, val_fold in zip(train_folds, val_folds):
        state_path = get_state_path(train_fold)
        val_dataset = BengaliDatasetTrain(
            folds=val_fold,
            aug=A.Compose([
                A.Resize(IMG_HEIGHT, IMG_WEIGHT, always_apply=True),
                A.Normalize(**STAT)
            ]),
            use_rgb=USE_RGB
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=VAL_BATCH_SIZE,
            shuffle=False,
            pin_memory=True
        )
        print(state_path)
        print('val_folds: ', val_fold)

        model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True, use_rgb=USE_RGB)
        model.to(DEVICE)
        model.load_state_dict(torch.load(state_path))

        val_loss, val_macro_recall = check_accuracy(val_loader, model, loss_fn)
        print(f'Val Loss: {val_loss:.4f} Macro recall: {val_macro_recall:.4f}')


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
        val_macro_recall = macro_recall(final_outputs, final_targets, verbose=True)
        val_loss = running_loss / len(loader)
        return val_loss, val_macro_recall


def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, history=None, num_epochs=25):
    state_path = get_state_path(train_loader.dataset.folds)
    if not os.path.isfile(state_path):
        print(f'Checking accuracy of {state_path}')
        model.load_state_dict(torch.load(state_path), strict=False)
        best_macro_recall = check_accuracy(val_loader, model, criterion)[1]
    else:
        best_macro_recall = 0.0
    print(f'initial recall={best_macro_recall}')

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

        train_macro_recall = macro_recall(final_outputs, final_targets, verbose=True)
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
            torch.save(model.state_dict(), state_path)
            best_macro_recall = val_macro_recall

        print()

    print(f'Best val Macro recall: {best_macro_recall:4f}')

    # load best model weights
    model.load_state_dict(torch.load(state_path))
    return model, history


def plot_graph(history, epoch):
    val_loss, val_recall = history['val_loss'], history['val_recall']
    train_loss, train_recall = history['train_loss'], history['train_recall']
    epochsx = np.arange(epoch)

    for metric, values in zip(['loss', 'recall'], ((val_loss, train_loss), (val_recall, train_recall))):
        plt.subplot(3, 1, 1)
        plt.title(f'{metric} on train and validation sets')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.plot(epochsx, values[0], '-^', label=f'val {metric}')
        plt.plot(epochsx, values[1], '-o', label=f'train {metric}')
        plt.legend()
        plt.grid(True)

    plt.show()


if __name__ == '__main__':
    main()
