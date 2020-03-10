export DEVICE="cuda"
export IMG_HEIGHT=68
export IMG_WIDTH=118
export EPOCH=80
export TRAIN_BATCH_SIZE=64
export TEST_BATCH_SIZE=16
export BASE_MODEL="resnet18"
export TRAINING_FOLDS_CSV="../input/train_folds.csv"
export PRELOAD_DATASET=1
export DEBUG=0


export TRAINING_FOLDS="(0,1,2,3)"
export VALIDATION_FOLDS="(4,)"
python3 src/train.py

export TRAINING_FOLDS="(0,1,2,4)"
export VALIDATION_FOLDS="(3,)"
python3 src/train.py

export TRAINING_FOLDS="(0,1,3,4)"
export VALIDATION_FOLDS="(2,)"
python3 src/train.py

export TRAINING_FOLDS="(0,2,3,4)"
export VALIDATION_FOLDS="(1,)"
python3 src/train.py

export TRAINING_FOLDS="(1,2,3,4)"
export VALIDATION_FOLDS="(0,)"
python3 src/train.py