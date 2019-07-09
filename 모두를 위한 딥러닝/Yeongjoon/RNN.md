
https://wdprogrammer.tistory.com/22?category=817754

```python
import keras
import numpy as np
import matplotlib.pyplot as plt
```

    c:\users\yeongjoon\appdata\local\continuum\anaconda3\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    


```python
timesteps = 100
input_features = 32
output_features = 64

inputs = np.random.random((timesteps, input_features))
state_t = np.zeros((output_features,))

# 입력에 대한 가중치
W = np.random.random((output_features, input_features))
# 이전 상태에 대한 가중치
U = np.random.random((output_features, output_features))
# 편향
b = np.random.random((output_features,))

successive_outputs = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t

final_output_sequence = np.stack(successive_outputs, axis = 0)
```


```python
final_output_sequence.shape
```




    (100, 64)




```python
from keras.layers import SimpleRNN, Embedding
from keras.models import Sequential
```


```python
# input_features == 10000
# output_features == 32

# 마지막 타임스텝의 출력만 반환
model1 = Sequential()
model1.add(Embedding(10000, 32))
model1.add(SimpleRNN(32))
model1.summary()

# 각 타임스텝의 출력을 모은 전체 시퀀스 반환
model2 = Sequential()
model2.add(Embedding(10000, 32))
model2.add(SimpleRNN(32, return_sequences=True))
model2.summary()

# 
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, None, 32)          320000    
    _________________________________________________________________
    simple_rnn_1 (SimpleRNN)     (None, 32)                2080      
    =================================================================
    Total params: 322,080
    Trainable params: 322,080
    Non-trainable params: 0
    _________________________________________________________________
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, None, 32)          320000    
    _________________________________________________________________
    simple_rnn_2 (SimpleRNN)     (None, None, 32)          2080      
    =================================================================
    Total params: 322,080
    Trainable params: 322,080
    Non-trainable params: 0
    _________________________________________________________________
    


```python
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000
maxlen = 500
batch_size = 32

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)

# restore np.load for future normal usage
np.load = np_load_old

print(len(input_train), '훈련 시퀀스')
print(len(input_test), '테스트 시퀀스')
print('시퀀스 패딩 (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train 크기:', input_train.shape)
print('input_test 크기:', input_test.shape)
print('y_train 크기:', y_train.shape)
```

    25000 훈련 시퀀스
    25000 테스트 시퀀스
    시퀀스 패딩 (samples x time)
    input_train 크기: (25000, 500)
    input_test 크기: (25000, 500)
    y_train 크기: (25000,)
    


```python
input_train[0]
```




    array([   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    1,   14,   22,   16,
             43,  530,  973, 1622, 1385,   65,  458, 4468,   66, 3941,    4,
            173,   36,  256,    5,   25,  100,   43,  838,  112,   50,  670,
              2,    9,   35,  480,  284,    5,  150,    4,  172,  112,  167,
              2,  336,  385,   39,    4,  172, 4536, 1111,   17,  546,   38,
             13,  447,    4,  192,   50,   16,    6,  147, 2025,   19,   14,
             22,    4, 1920, 4613,  469,    4,   22,   71,   87,   12,   16,
             43,  530,   38,   76,   15,   13, 1247,    4,   22,   17,  515,
             17,   12,   16,  626,   18,    2,    5,   62,  386,   12,    8,
            316,    8,  106,    5,    4, 2223, 5244,   16,  480,   66, 3785,
             33,    4,  130,   12,   16,   38,  619,    5,   25,  124,   51,
             36,  135,   48,   25, 1415,   33,    6,   22,   12,  215,   28,
             77,   52,    5,   14,  407,   16,   82,    2,    8,    4,  107,
            117, 5952,   15,  256,    4,    2,    7, 3766,    5,  723,   36,
             71,   43,  530,  476,   26,  400,  317,   46,    7,    4,    2,
           1029,   13,  104,   88,    4,  381,   15,  297,   98,   32, 2071,
             56,   26,  141,    6,  194, 7486,   18,    4,  226,   22,   21,
            134,  476,   26,  480,    5,  144,   30, 5535,   18,   51,   36,
             28,  224,   92,   25,  104,    4,  226,   65,   16,   38, 1334,
             88,   12,   16,  283,    5,   16, 4472,  113,  103,   32,   15,
             16, 5345,   19,  178,   32])



+ 첫번째 인자(input_dim) : 단어 사전의 크기를 말하며 총 20,000개의 단어 종류가 있다는 의미입니다. 이 값은 앞서 imdb.load_data() 함수의 num_words 인자값과 동일해야 합니다.
+ 두번째 인자(output_dim) : 단어를 인코딩 한 후 나오는 벡터 크기 입니다. 이 값이 128이라면 단어를 128차원의 의미론적 기하공간에 나타낸다는 의미입니다. 단순하게 빈도수만으로 단어를 표시한다면, 10과 11은 빈도수는 비슷하지만 단어로 볼 때는 전혀 다른 의미를 가지고 있습니다. 하지만 의미론적 기하공간에서는 거리가 가까운 두 단어는 의미도 유사합니다. 즉 임베딩 레이어는 입력되는 단어를 의미론적으로 잘 설계된 공간에 위치시켜 벡터로 수치화 시킨다고 볼 수 있습니다.
+ input_length : 단어의 수 즉 문장의 길이를 나타냅니다. 임베딩 레이어의 출력 크기는 샘플 수 * output_dim * input_lenth가 됩니다. 임베딩 레이어 다음에 Flatten 레이어가 온다면 반드시 input_lenth를 지정해야 합니다. 플래튼 레이어인 경우 입력 크기가 알아야 이를 1차원으로 만들어서 Dense 레이어에 전달할 수 있기 때문입니다.


```python
from keras.layers import Dense
model = Sequential()
model.add(Embedding(max_features, 32)) # output = (time_steps, input_features, output_features)
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
```

    WARNING:tensorflow:From C:\Users\user5\Anaconda3\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Train on 20000 samples, validate on 5000 samples
    Epoch 1/10
    20000/20000 [==============================] - 16s 816us/step - loss: 0.6425 - acc: 0.6137 - val_loss: 0.5040 - val_acc: 0.7718
    Epoch 2/10
    20000/20000 [==============================] - 16s 792us/step - loss: 0.4114 - acc: 0.8257 - val_loss: 0.4667 - val_acc: 0.7962
    Epoch 3/10
    20000/20000 [==============================] - 16s 777us/step - loss: 0.2966 - acc: 0.8815 - val_loss: 0.4723 - val_acc: 0.7816
    Epoch 4/10
    20000/20000 [==============================] - 16s 791us/step - loss: 0.2248 - acc: 0.9130 - val_loss: 0.4046 - val_acc: 0.8292
    Epoch 5/10
    20000/20000 [==============================] - 16s 793us/step - loss: 0.1715 - acc: 0.9371 - val_loss: 0.4211 - val_acc: 0.8314
    Epoch 6/10
    20000/20000 [==============================] - 17s 854us/step - loss: 0.1126 - acc: 0.9613 - val_loss: 0.4066 - val_acc: 0.8592
    Epoch 7/10
    20000/20000 [==============================] - 18s 876us/step - loss: 0.0736 - acc: 0.9768 - val_loss: 0.6656 - val_acc: 0.7662
    Epoch 8/10
    20000/20000 [==============================] - 16s 795us/step - loss: 0.0450 - acc: 0.9872 - val_loss: 0.5122 - val_acc: 0.8470
    Epoch 9/10
    20000/20000 [==============================] - 16s 791us/step - loss: 0.0275 - acc: 0.9916 - val_loss: 0.8576 - val_acc: 0.77260.991
    Epoch 10/10
    20000/20000 [==============================] - 19s 926us/step - loss: 0.0184 - acc: 0.9944 - val_loss: 0.7568 - val_acc: 0.7840
    


```python
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```


![png](output_9_0.png)



![png](output_9_1.png)



```python
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
```

    Train on 20000 samples, validate on 5000 samples
    Epoch 1/10
    20000/20000 [==============================] - 43s 2ms/step - loss: 0.5149 - acc: 0.7591 - val_loss: 0.4039 - val_acc: 0.8486
    Epoch 2/10
    20000/20000 [==============================] - 43s 2ms/step - loss: 0.2923 - acc: 0.8868 - val_loss: 0.3064 - val_acc: 0.8786
    Epoch 3/10
    20000/20000 [==============================] - 42s 2ms/step - loss: 0.2360 - acc: 0.9100 - val_loss: 0.3147 - val_acc: 0.8862
    Epoch 4/10
    20000/20000 [==============================] - 42s 2ms/step - loss: 0.1964 - acc: 0.9291 - val_loss: 0.3113 - val_acc: 0.8664
    Epoch 5/10
    20000/20000 [==============================] - 42s 2ms/step - loss: 0.1736 - acc: 0.9379 - val_loss: 0.2823 - val_acc: 0.8832
    Epoch 6/10
    20000/20000 [==============================] - 48s 2ms/step - loss: 0.1534 - acc: 0.9443 - val_loss: 0.4010 - val_acc: 0.8746
    Epoch 7/10
    20000/20000 [==============================] - 47s 2ms/step - loss: 0.1434 - acc: 0.9491 - val_loss: 0.3511 - val_acc: 0.8848
    Epoch 8/10
    20000/20000 [==============================] - 46s 2ms/step - loss: 0.1261 - acc: 0.9560 - val_loss: 0.3412 - val_acc: 0.8888
    Epoch 9/10
    20000/20000 [==============================] - 47s 2ms/step - loss: 0.1148 - acc: 0.9603 - val_loss: 0.3697 - val_acc: 0.8858
    Epoch 10/10
    20000/20000 [==============================] - 47s 2ms/step - loss: 0.1053 - acc: 0.9634 - val_loss: 0.4344 - val_acc: 0.8314
    


```python
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```


![png](output_11_0.png)



![png](output_11_1.png)


### 살펴볼 것

**순환 드롭아웃(recurrent dropout)** : 순환 층에서 과대적합을 방지

**스태킹 순환 층(stacking recurrent layer)** : 네트워크의 표현 능력을 증가시킴

**양방향 순환 층(bidirectional recurrent layer)** : 순환 네트워크에 같은 정보를 다른 방향으로 주입하여 정확도를 높이고 기억을 더 오래 유지시킴

*날씨 시계열 데이터셋 사용* : 독일 예나(Jena) 시에 있는 막스 플랑크 생물지구화학 연구소의 지상관측소에서 수집한 데이터로, 14개의 기온, 기압, 습도, 풍향 등의 관측치가 10분마다 기록되어 있다.

https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip


```python
import pandas as pd
jena_climate = pd.read_csv('jena_climate_2009_2016.csv')
jena_climate = jena_climate.drop('Date Time', axis=1)
print('data 개수: {}'.format(len(jena_climate)))
print('<header>\n{}'.format(jena_climate.columns))
jena_climate.head(5)
```

    data 개수: 420551
    <header>
    Index(['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
           'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
           'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
           'wd (deg)'],
          dtype='object')
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>p (mbar)</th>
      <th>T (degC)</th>
      <th>Tpot (K)</th>
      <th>Tdew (degC)</th>
      <th>rh (%)</th>
      <th>VPmax (mbar)</th>
      <th>VPact (mbar)</th>
      <th>VPdef (mbar)</th>
      <th>sh (g/kg)</th>
      <th>H2OC (mmol/mol)</th>
      <th>rho (g/m**3)</th>
      <th>wv (m/s)</th>
      <th>max. wv (m/s)</th>
      <th>wd (deg)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>996.52</td>
      <td>-8.02</td>
      <td>265.40</td>
      <td>-8.90</td>
      <td>93.3</td>
      <td>3.33</td>
      <td>3.11</td>
      <td>0.22</td>
      <td>1.94</td>
      <td>3.12</td>
      <td>1307.75</td>
      <td>1.03</td>
      <td>1.75</td>
      <td>152.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>996.57</td>
      <td>-8.41</td>
      <td>265.01</td>
      <td>-9.28</td>
      <td>93.4</td>
      <td>3.23</td>
      <td>3.02</td>
      <td>0.21</td>
      <td>1.89</td>
      <td>3.03</td>
      <td>1309.80</td>
      <td>0.72</td>
      <td>1.50</td>
      <td>136.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>996.53</td>
      <td>-8.51</td>
      <td>264.91</td>
      <td>-9.31</td>
      <td>93.9</td>
      <td>3.21</td>
      <td>3.01</td>
      <td>0.20</td>
      <td>1.88</td>
      <td>3.02</td>
      <td>1310.24</td>
      <td>0.19</td>
      <td>0.63</td>
      <td>171.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>996.51</td>
      <td>-8.31</td>
      <td>265.12</td>
      <td>-9.07</td>
      <td>94.2</td>
      <td>3.26</td>
      <td>3.07</td>
      <td>0.19</td>
      <td>1.92</td>
      <td>3.08</td>
      <td>1309.19</td>
      <td>0.34</td>
      <td>0.50</td>
      <td>198.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>996.51</td>
      <td>-8.27</td>
      <td>265.15</td>
      <td>-9.04</td>
      <td>94.1</td>
      <td>3.27</td>
      <td>3.08</td>
      <td>0.19</td>
      <td>1.92</td>
      <td>3.09</td>
      <td>1309.00</td>
      <td>0.32</td>
      <td>0.63</td>
      <td>214.3</td>
    </tr>
  </tbody>
</table>
</div>



각 데이터는 시간 순서대로 나타나있으므로 각 row가 하나의 타임스텝이며 column은 input feature다.


```python
float_data = jena_climate.values

# 온도
temp = float_data[:, 1]
plt.plot(range(len(temp)), temp)
plt.show()
```


![png](output_16_0.png)



```python
plt.plot(range(1440), temp[:1440])
```




    [<matplotlib.lines.Line2D at 0x15c0a28d9b0>]




![png](output_17_1.png)


+ 위는 처음 10일 간의 온도 데이터 그래프다.
+ 10분마다 하나의 데이터가 있으므로 1시간에 6개의 데이터, 1일에 24x6=144개의 데이터, 10일에는 1440개의 데이터가 있다.

연간 데이터 주기성이 안정적이기 때문에 지난 몇 달간 데이터를 사용하여 다음 달의 평균 온도를 예측하는 문제는 쉬운 편이다.

**하지만 일별 데이터를 살펴보면 온도 변화는 매우 불안정하다. 일자별 수준의 시계열 데이터를 예측할 수 있을까?**

### 1.데이터 정규화


```python
mean = float_data[:200000].mean(axis=0) # 처음 20만개의 데이터만 사용
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std
```

### 2. 데이터 제네레이터 생성

+ lookback : 얼마나 이전의 데이터로부터 학습할 것인가 - 어느 지점을 예측하기 위해 학습되는 데이터의 시작점을 결정
+ steps : 데이터를 어느 정도 간격으로 수집할 것인가 - 6으로 하면 여기선 1시간마다 데이터 포인터를 샘플링하게 된다.
+ delay : 얼마 뒤의 데이터를 예측할 것인가 - 예측할 미래의 지점을 결정


```python
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    
    # rows는 현재 지점(기준점)의 samples라고 생각하면 됨
    # batch_size만큼 지점을 선택
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        
        # lookback // step : 학습될 타임스텝 수
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
```

### 3. 학습 준비


```python
lookback = 1440  # 10일 -> 10일 간의 데이터를 학습
step = 6  # 1시간
delay = 144  # 1일 -> 하루 뒤 온도를 예측
batch_size = 128

train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, max_index=200000, shuffle=True, step=step, batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=200001, max_index=300000, step=step, batch_size=batch_size)
test_gen = generator(float_data, lookback=lookback, delay=delay, min_index=300001, max_index=None, step=step, batch_size=batch_size)
# 전체 검증 세트를 순회하기 위해 val_gen에서 추출할 횟수 
val_steps = (300000 - 200001 - lookback) // batch_size
```

### 4. 모델 구축 및 학습

GRU layer는 LSTM과 같은 원리로 작동하지만 조금 더 간결하고 계산 비용이 덜 든다.


```python
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20, validation_data=val_gen, validation_steps=val_steps)
```

    Epoch 1/20
    500/500 [==============================] - 90s 181ms/step - loss: 0.3032 - val_loss: 0.2766
    Epoch 2/20
    500/500 [==============================] - 85s 170ms/step - loss: 0.2857 - val_loss: 0.2682
    Epoch 3/20
    500/500 [==============================] - 98s 197ms/step - loss: 0.2787 - val_loss: 0.2638
    Epoch 4/20
    500/500 [==============================] - 100s 200ms/step - loss: 0.2753 - val_loss: 0.2605
    Epoch 5/20
    500/500 [==============================] - 97s 193ms/step - loss: 0.2679 - val_loss: 0.2623
    Epoch 6/20
    500/500 [==============================] - 93s 187ms/step - loss: 0.2608 - val_loss: 0.2641
    Epoch 7/20
    500/500 [==============================] - 92s 185ms/step - loss: 0.2596 - val_loss: 0.2627
    Epoch 8/20
    500/500 [==============================] - 97s 194ms/step - loss: 0.2554 - val_loss: 0.2668
    Epoch 9/20
    500/500 [==============================] - 97s 194ms/step - loss: 0.2515 - val_loss: 0.2648
    Epoch 10/20
    500/500 [==============================] - 100s 199ms/step - loss: 0.2466 - val_loss: 0.2716
    Epoch 11/20
    500/500 [==============================] - 98s 196ms/step - loss: 0.2425 - val_loss: 0.2686
    Epoch 12/20
    500/500 [==============================] - 91s 182ms/step - loss: 0.2401 - val_loss: 0.2754
    Epoch 13/20
    500/500 [==============================] - 98s 197ms/step - loss: 0.2343 - val_loss: 0.2762
    Epoch 14/20
    500/500 [==============================] - 98s 197ms/step - loss: 0.2341 - val_loss: 0.2782
    Epoch 15/20
    500/500 [==============================] - 95s 189ms/step - loss: 0.2303 - val_loss: 0.2772
    Epoch 16/20
    500/500 [==============================] - 93s 186ms/step - loss: 0.2269 - val_loss: 0.2866
    Epoch 17/20
    500/500 [==============================] - 96s 192ms/step - loss: 0.2227 - val_loss: 0.2859
    Epoch 18/20
    224/500 [============>.................] - ETA: 38s - loss: 0.2227


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-24-8f8b2090cdb1> in <module>
          7 model.add(layers.Dense(1))
          8 model.compile(optimizer=RMSprop(), loss='mae')
    ----> 9 history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20, validation_data=val_gen, validation_steps=val_steps)
    

    ~\Anaconda3\lib\site-packages\keras\legacy\interfaces.py in wrapper(*args, **kwargs)
         89                 warnings.warn('Update your `' + object_name + '` call to the ' +
         90                               'Keras 2 API: ' + signature, stacklevel=2)
    ---> 91             return func(*args, **kwargs)
         92         wrapper._original_function = func
         93         return wrapper
    

    ~\Anaconda3\lib\site-packages\keras\engine\training.py in fit_generator(self, generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)
       1416             use_multiprocessing=use_multiprocessing,
       1417             shuffle=shuffle,
    -> 1418             initial_epoch=initial_epoch)
       1419 
       1420     @interfaces.legacy_generator_methods_support
    

    ~\Anaconda3\lib\site-packages\keras\engine\training_generator.py in fit_generator(model, generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)
        215                 outs = model.train_on_batch(x, y,
        216                                             sample_weight=sample_weight,
    --> 217                                             class_weight=class_weight)
        218 
        219                 outs = to_list(outs)
    

    ~\Anaconda3\lib\site-packages\keras\engine\training.py in train_on_batch(self, x, y, sample_weight, class_weight)
       1215             ins = x + y + sample_weights
       1216         self._make_train_function()
    -> 1217         outputs = self.train_function(ins)
       1218         return unpack_singleton(outputs)
       1219 
    

    ~\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py in __call__(self, inputs)
       2713                 return self._legacy_call(inputs)
       2714 
    -> 2715             return self._call(inputs)
       2716         else:
       2717             if py_any(is_tensor(x) for x in inputs):
    

    ~\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py in _call(self, inputs)
       2673             fetched = self._callable_fn(*array_vals, run_metadata=self.run_metadata)
       2674         else:
    -> 2675             fetched = self._callable_fn(*array_vals)
       2676         return fetched[:len(self.outputs)]
       2677 
    

    ~\Anaconda3\lib\site-packages\tensorflow\python\client\session.py in __call__(self, *args, **kwargs)
       1437           ret = tf_session.TF_SessionRunCallable(
       1438               self._session._session, self._handle, args, status,
    -> 1439               run_metadata_ptr)
       1440         if run_metadata:
       1441           proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)
    

    KeyboardInterrupt: 



```python
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```

### 5. 순환 드롭아웃 사용

**일반적인 Dropout**

\begin{split}\begin{aligned}
    h & = \sigma(W_1 x + b_1) \\
    o & = W_2 h + b_2 \\
    \hat{y} & = \mathrm{softmax}(o)
\end{aligned}\end{split}

**LSTM Dropout**

http://nmhkahn.github.io/RNN-Regularizations

순환 신경망은 일반적인 드롭아웃처럼 적용하면 오히려 학습에 방해가 된다.

타임스텝마다 랜덤하게 드롭아웃 마스크를 바꾸는 것이 아닌 동일한 드롭아웃 마스크(동일한 유닛의 드롭 패턴)를 모든 타임스텝에 적용해야 한다.

즉, 규제를 위해서는 순환 층 내부 계산에 사용된 활성화 함수에 타임스텝마다 동일한 드롭아웃 마스크를 적용해야 한다.

![](http://nmhkahn.github.io/assets/RNN-Reg/p1-dropout.png)

위의 점선이 dropout이 적용된 것이고 실선은 적용되지 않은 것 => **현재 timestep에서 들어온 입력값 혹은 이전 레이어의 값만 dropout**하는 의미다.


```python
model = Sequential()
model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40, validation_data=val_gen, validation_steps=val_steps)
```
