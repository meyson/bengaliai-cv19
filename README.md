# Bengali.AI Handwritten Grapheme Classification Pytorch

[Original competition](https://www.kaggle.com/c/bengaliai-cv19)


Test recall calculated according to [competition's evaluation metric](https://www.kaggle.com/c/bengaliai-cv19/overview/evaluation)

## How to run training?
```bash
git clone https://github.com/meyson/bengaliai-cv19.git
cd bengaliai-cv19
kaggle competitions download -c bengaliai-cv19
unzip bengaliai-cv19.zip -d data
python3 create_folds.py
python3 create_train_images_png.py
chmod +x run.sh && ./run.sh
```

| Name      | # layers | # params| Test recall|
|-----------|---------:|--------:|:---------------------:|
|[SqueezeNet](https://github.com/meyson/bengaliai-cv19/blob/master/pretrained_models)   |    -    | 1.3M   | ~0.95|
|[EfficientNet B3](https://github.com/meyson/bengaliai-cv19/blob/master/pretrained_models)   |    -    | 1.3M   | ~0.965|