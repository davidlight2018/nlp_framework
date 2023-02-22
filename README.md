# NLP深度学习训练框架



## 数据集测试结果

### 1. cluener

训练集 10748，验证集 1343

#### a. 模型效果

训练 5 epochs

|         模型         | valid_precision | valid_recall | valid_f1 |
| :------------------: | :-------------: | :----------: | :------: |
|      bert-base       |     0.7349      |    0.8040    |  0.7679  |
|    bert-base+CRF     |     0.7488      |    0.8092    |  0.7778  |
| bert-base+biLSTM+CRF |     0.7551      |    0.8001    |  0.7770  |

#### b. 各个实体效果

|     实体      | bert-base | bert-base+CRF | bert-base+biLSTM+CRF | Human Performance |
| :-----------: | :-------: | :-----------: | :------------------: | :---------------: |
|    address    |   61.42   |     64.05     |        61.18         |                   |
|     book      |   78.23   |     79.25     |        79.87         |                   |
|    company    |   76.49   |     77.66     |        78.93         |                   |
|     game      |   82.32   |     81.28     |        81.98         |                   |
|  government   |   80.75   |     81.37     |        80.83         |                   |
|     movie     |   77.89   |     78.43     |        82.51         |                   |
|     name      |   85.89   |     86.86     |        86.65         |                   |
| organization  |   75.03   |     77.89     |        79.24         |                   |
|   position    |   78.72   |     78.22     |        77.94         |                   |
|     scene     |   70.18   |     71.03     |        67.30         |                   |
| Overall@Micro |   76.79   |     77.78     |        77.70         |                   |

#### c. 显卡训练速度

模型: bert-base + CRF

|      显卡      | train_step_time | train_epoch_time | valid_step_time | valid_epoch_time |
| :------------: | :-------------: | :--------------: | :-------------: | :--------------: |
|    RTX 2070    |      182ms      |      1m21s       |     63.1ms      |        3s        |
| M1 Air (7 gpu) |      1.1s       |      8m15s       |     352.1ms     |       19s        |

#### 

### 2. 人民日报 2014

训练集 229016，验证集 57253

#### a. 模型效果

训练 5 epochs

|             模型             | valid_acc | valid_recall | valid_f1 |  loss  |
| :--------------------------: | :-------: | :----------: | :------: | :----: |
| bert-base-chinese+biLSTM+CRF |  0.9845   |    0.9869    |  0.9857  | 0.7369 |

#### b. 各个实体效果

|     实体      | bert-base-chinese+CRF | bert-base | roberta-wwm-large-ext | Human Performance |
| :-----------: | :-------------------: | :-------: | :-------------------: | :---------------: |
|      LOC      |         98.86         |           |                       |                   |
|      ORG      |         99.21         |           |                       |                   |
|      PER      |         97.24         |           |                       |                   |
|       T       |         99.51         |           |                       |                   |
| Overall@Micro |         98.57         |           |                       |                   |

#### c. 显卡训练速度

M1 Air 16G 内存不够，无法完整利用 GPU

|      显卡      | train_step_time | train_epoch_time | valid_step_time | valid_epoch_time |
| :------------: | :-------------: | :--------------: | :-------------: | :--------------: |
|    RTX 3090    |     399.1ms     |     1h03m28s     |     157.4ms     |      6m15s       |
|    RTX 2070    |     510.7ms     |     2h01m50s     |     223.0ms     |      13m18s      |
| M1 Air (7 gpu) |        -        |        -         |        -        |        -         |



### 3. CCF 2020

训练集 2012，验证集 503

#### a. 模型效果

训练 5 epochs

|         模型          | valid_acc | valid_recall | valid_f1 |  loss   |
| :-------------------: | :-------: | :----------: | :------: | :-----: |
| bert-base-chinese+CRF |  0.7219   |    0.7501    |  0.7357  | 41.2741 |

#### b. 各个实体效果

|     实体      | bert-base-chinese+CRF | bert-base | roberta-wwm-large-ext | Human Performance |
| :-----------: | :-------------------: | :-------: | :-------------------: | :---------------: |
|     Name      |         81.18         |           |                       |                   |
| Organization  |         71.86         |           |                       |                   |
|   Position    |         72.03         |           |                       |                   |
|    Company    |         74.48         |           |                       |                   |
|    Address    |         57.77         |           |                       |                   |
|     Game      |         80.91         |           |                       |                   |
|  Government   |         75.95         |           |                       |                   |
|     Scene     |         64.29         |           |                       |                   |
|     Book      |         71.84         |           |                       |                   |
|     Movie     |         78.44         |           |                       |                   |
| Overall@Macro |         73.57         |           |                       |                   |

