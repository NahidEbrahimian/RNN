### 10- Video_Classification_CRNN

- Video classificatio nusing CRNN on ucf101_top5 dataset

#### Model

Backbone: `my vgg base model` for feature extraction

RNN modules: RNN, GRU are tested

The performance of GRU module was better than RNN madules

#

#### Dataset

- ucf101_top5 dataset containing 573 video from 5 classes

#

#### Result

| Model | Val Accuracy |
| :---         |     :---:      |
| RNN Model  | 0.87   |
|GRU Model    | 0.94     |
