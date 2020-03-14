# Bengali.AI Handwritten Grapheme Classification Pytorch (SqueezeNet)

[Original competition](https://www.kaggle.com/c/bengaliai-cv19)


Test recall calculated according to [competition's evaluation metric](https://www.kaggle.com/c/bengaliai-cv19/overview/evaluation)

## How to run training?
```bash
git clone https://github.com/meyson/bengaliai-cv19.git
cd bengaliai-cv19
python3 create_train_images_png.py
python3 create_folds.py
chmod +x run.sh && ./run.sh
```

| Name      | # layers | # params| Test recall|
|-----------|---------:|--------:|:---------------------:|
|[SqueezeNet](https://github.com/meyson/bengaliai-cv19/blob/master/pretrained_models/squeezenet_train_folds_(0%2C%201%2C%202%2C%203).h5)   |    -    | 1.3M   | 0.95+|