# NLP深度学习训练框架



## 数据集测试结果

### 1. cluener

训练集 10748，验证集 1343

#### a. 显卡训练速度

|      显卡      | train_step_time | train_epoch_time | valid_step_time | valid_epoch_time |
| :------------: | :-------------: | :--------------: | :-------------: | :--------------: |
|    RTX 2070    |      182ms      |      1m21s       |     63.1ms      |        3s        |
| M1 Air (7 gpu) |      1.1s       |      8m15s       |     352.1ms     |       19s        |

#### b. 模型效果

训练 5 epochs

|         模型          | valid_acc | valid_recall | valid_f1 |  loss  |
| :-------------------: | :-------: | :----------: | :------: | :----: |
| bert-base-chinese+CRF |  0.7977   |    0.8008    |  0.7992  | 8.5201 |

#### c. 各个实体效果

|     实体      | bert-base-chinese+CRF | bert-base | roberta-wwm-large-ext | Human Performance |
| :-----------: | :-------------------: | :-------: | :-------------------: | :---------------: |
|  Person Name  |         87.75         |           |                       |                   |
| Organization  |         78.58         |           |                       |                   |
|   Position    |         80.60         |           |                       |                   |
|    Company    |         79.69         |           |                       |                   |
|    Address    |         63.71         |           |                       |                   |
|     Game      |         85.67         |           |                       |                   |
|  Government   |         83.53         |           |                       |                   |
|     Scene     |         72.86         |           |                       |                   |
|     Book      |         82.67         |           |                       |                   |
|     Movie     |         86.87         |           |                       |                   |
| Overall@Macro |         79.92         |           |                       |                   |



### 2. 人民日报2014

训练集 229016，验证集 57253

#### a. 显卡训练速度

M1 Air 16G 内存不够，无法完整利用 GPU

|      显卡      | train_step_time | train_epoch_time | valid_step_time | valid_epoch_time |
| :------------: | :-------------: | :--------------: | :-------------: | :--------------: |
|    RTX 2070    |     510.7ms     |     2h01m50s     |     223.0ms     |      13m18s      |
| M1 Air (7 gpu) |        -        |        -         |        -        |        -         |

#### b. 模型效果

训练 5 epochs

|         模型          | valid_acc | valid_recall | valid_f1 |  loss  |
| :-------------------: | :-------: | :----------: | :------: | :----: |
| bert-base-chinese+CRF |  0.9862   |    0.9870    |  0.9866  | 0.6073 |

#### c. 各个实体效果

|     实体      | bert-base-chinese+CRF | bert-base | roberta-wwm-large-ext | Human Performance |
| :-----------: | :-------------------: | :-------: | :-------------------: | :---------------: |
|      LOC      |         98.85         |           |                       |                   |
|      ORG      |         99.31         |           |                       |                   |
|      PER      |         97.54         |           |                       |                   |
|       T       |         99.53         |           |                       |                   |
| Overall@Macro |         98.66         |           |                       |                   |
