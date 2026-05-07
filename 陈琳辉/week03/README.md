# Week 03 - RNN/LSTM 文本分类实验

## 任务描述

使用中文酒店评论数据集（ChnSentiCorp_htl_all.csv）进行二分类任务，分别实现 RNN 和 LSTM 模型的训练和评估。

## 数据集

- **文件**: `ChnSentiCorp_htl_all.csv`
- **规模**: 7767 条评论
- **标签分布**:
  - 正面评论 (label=1): 5322 条
  - 负面评论 (label=0): 2445 条
- **特点**: 餐厅、酒店评论，情感分类

## 模型架构

### 基础模型: Embedding → RNN/LSTM → MaxPooling → Dense → Sigmoid

```
Embedding Layer (词嵌入层)
        ↓
  RNN/LSTM Layer (双向)
        ↓
  Max Pooling (序列维度)
        ↓
  BatchNorm + Dropout
        ↓
  Dense Layer → Sigmoid → Output
```

### RNN vs LSTM

| 特性 | RNN | LSTM |
|------|-----|------|
| 门控机制 | 无 | 输入门、遗忘门、输出门 |
| 长距离依赖 | 难以捕捉 | 有效缓解梯度消失 |
| 参数量 | 较少 | 较多 |
| 训练速度 | 快 | 较慢 |

## 使用方法

### 1. 基本训练

```bash
# 训练 RNN 模型
cd 陈琳辉/week03
python week03_rnn_lstm_train.py --model rnn

# 训练 LSTM 模型
python week03_rnn_lstm_train.py --model lstm
```

### 2. 自定义参数

```bash
# 自定义超参数
python week03_rnn_lstm_train.py \
    --model lstm \
    --epochs 30 \
    --hidden 128 \
    --embed 128 \
    --batch_size 32 \
    --lr 1e-3 \
    --dropout 0.3 \
    --num_layers 2
```

### 3. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | lstm | 模型类型: rnn / lstm |
| `--epochs` | 20 | 训练轮数 |
| `--batch_size` | 64 | 批次大小 |
| `--hidden` | 64 | RNN/LSTM 隐藏层维度 |
| `--embed` | 128 | Embedding 维度 |
| `--lr` | 1e-3 | 学习率 |
| `--dropout` | 0.3 | Dropout 比率 |
| `--maxlen` | 128 | 最大序列长度 |
| `--num_layers` | 1 | RNN/LSTM 层数 |
| `--save_model` | False | 是否保存模型 |

## 输出文件

训练完成后会在 `output/` 目录下生成:

- `rnn_training_curves.png` - RNN 训练曲线
- `lstm_training_curves.png` - LSTM 训练曲线
- `rnn_history.json` - RNN 训练历史
- `lstm_history.json` - LSTM 训练历史
- `best_rnn_model.pt` - 最佳 RNN 模型 (需加 --save_model)
- `best_lstm_model.pt` - 最佳 LSTM 模型 (需加 --save_model)

## 参考代码说明

参考文件夹 `week03/参考/` 包含以下示例:

| 文件 | 说明 |
|------|------|
| `train_chinese_cls_rnn.py` | RNN 文本分类完整示例 |
| `RNNforward.py` | RNN 前向计算原理与手动实现 |
| `LSTMforward.py` | LSTM 前向计算原理与手动实现 |
| `embedding_padding_demo.py` | Embedding 和 Padding 教学 |
| `Pooling.py` | Pooling 层使用示例 |
| `BatchNorm.py` | BatchNorm 层使用示例 |
| `Dropout.py` | Dropout 层使用示例 |
| `adam_demo.py` | Adam 优化器原理与手动实现 |

## 实验建议

1. **对比实验**: 分别运行 RNN 和 LSTM，对比收敛速度和最终性能
2. **调参实验**: 尝试不同的 hidden_dim、embed_dim、num_layers
3. **正则化**: 调整 dropout 比率防止过拟合
4. **学习率调度**: 观察 ReduceLROnPlateau 的效果
5. **多轮训练**: 增加 epochs 观察是否继续收敛

## 预期结果

- 基础模型在验证集上应达到 **80%-90%** 的准确率
- LSTM 通常比 RNN 表现更好（尤其在长文本上）
- 双向模型比单向模型效果更好

## 依赖

```bash
pip install torch numpy matplotlib scikit-learn pandas
```
