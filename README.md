# RNN


### 01- RNN_Classification

Simple RNN training for classification task of 3 signal: Sine, Square, Triangle.

---

### 02- RNN_Regression

Simple RNN training for sine wave estimation.

![download](https://user-images.githubusercontent.com/82975802/161122428-711b0824-f819-40c4-92ab-ffa7384ce342.png)

---

### 03- RNN_vs_GRU_Classification

Comparison of RNN model and GRU model for classification task of 3 signal: Sine, Square and Triangle, after 100 epoch training.


| Model | Accuracy |
| :---         |     :---:      |
| RNN Model  | 0.9315   |
|GRU Model    | 0.9383     |

---

### 04- RNN_vs_GRU_Regression

Comparison of RNN model and GRU model for regression task of sine wave estimation after 100 epoch training.


| Model | loss |
| :---         |     :---:      |
| RNN Model  | 0.0027   |
|GRU Model    | 0.0026    |


![01](https://user-images.githubusercontent.com/82975802/161124533-97516304-d0e1-4b09-9889-48d259a5274a.png)


![02](https://user-images.githubusercontent.com/82975802/161124491-0d1061c0-c7e9-428a-98a7-77f93c762a71.png)

---

### 05- Ball_Move_Data_Generation

Generate data for ball move direction

![image](https://user-images.githubusercontent.com/82975802/163728473-e6681737-8077-464e-a9c3-17a9a3e40115.png)

---

### 06- GRU_Implementation_from_Scratch

GRU implementation from scratch + inference

---

### 07-LSTM_Implementation_from_Scrat

LSTM implementation from scratch + inference

---

### 08- Ball_Move_Direction_Classification

- Generate data for ball move direction 

- Classification of direction using RNN, GRU and LSTM

---

### 09- VideoClassificationCRNN


- [x] 09- Video_Classification_CRNN.ipynb(train)
- [x] inference.py
- [x] models.py (gru, lstm, rnn)
- [x] load_video.py
- [x] requirements.txt

# 

#### Model

Backbone: `ResNet50V2` and `my vgg base model` for feature extraction

RNN modules: RNN, GRU and LSTM are tested

The performance of GRU module was better than other madules

#

#### Dataset

Dataset contains videos from 2 classes

<br>
<br>

***Due to insufficient data, the training was not done well. but this project can be used for other video classification tasks using CRNNs***

---

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
