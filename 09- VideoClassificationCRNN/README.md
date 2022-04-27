## Video Classification using CRNNs


- [x] 09- Video_Classification_CRNN.ipynb(train)
- [x] inference.py
- [x] models.py (gru, lstm, rnn)
- [x] load_video.py
- [x] requirements.txt

# 

### Model

Backbone: `ResNet50V2` for feature extraction

RNN modules: RNN, GRU and LSTM are tested

The performance of GRU module was better than other madules

#

### Dataset

Dataset contains videos from 2 classes

#

### Train

1- First, you must install requirements

2- For trainig, you can run `Video_Classification_CRNN.ipynb` file

# 

### Inference

1- For inference, first you must install requirements

3- Then, run the folllowing command:

- `--input_path` is your input video path


```
python3 inference.py --input_path ./01.mp4
```

- Finally, The inference output is stored in the 'output' directory


#

***Due to insufficient data, the training was not done well. but this project can be used for other video classification tasks using CRNNs***
