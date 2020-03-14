export DEVICE="cuda"
export IMG_HEIGHT=137
export IMG_WEIGHT=236
export EPOCH=40
export TRAIN_BATCH_SIZE=32
export VAL_BATCH_SIZE=16
export BASE_MODEL="squeezenet"
export TRAINING_FOLDS_CSV="../input/train_folds.csv"
export PRELOAD_DATASET=0
# export CHECKPOINT="pretrained_models/.h5"


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