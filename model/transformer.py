import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import load_model
from load_dataset import train_dataset, val_dataset, test_dataset
import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 限制 TensorFlow 只使用第一个 GPU（如果有多张卡）
        tf.config.set_visible_devices(gpus[0], 'GPU')
        # 设置内存增长，避免一次性占用所有显存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"找到 {len(gpus)} 个 GPU，将使用: {gpus[0].name}")
    except RuntimeError as e:
        # 异常处理
        print(e)
else:
    print("未找到 GPU，将使用 CPU 进行训练。")

# Assuming 'train_dataset' is your tf.data.Dataset object
# Take one batch to inspect its structure and shape
for (mel_batch, coch_batch), labels_batch in train_dataset.take(1):
    print("--- Inspecting a single batch from the dataset ---")
    print("Mel input batch shape:", mel_batch.shape)
    print("Cochleagram input batch shape:", coch_batch.shape)
    print("Labels batch shape:", labels_batch.shape)
# --- 模型超参数定义 ---
# 输入形状
SEQUENCE_LENGTH = 60
MEL_SHAPE = (96, 44, 1)
COCH_SHAPE = (11, 44, 1)

# CNN 部分
CNN_FILTERS = 32
CNN_OUTPUT_DIM = 32

# Transformer 部分
TRANSFORMER_HEADS = 8
TRANSFORMER_UNITS = 32  # Transformer Feed-Forward部分的隐藏层大小
TRANSFORMER_LAYERS = 1  # Transformer Encoder的层数
MODEL_DIM = 64  # CNN融合后的特征维度，也是Transformer的输入维度

# 其他
DROPOUT_RATE = 0.2


def build_mel_cnn(input_shape=(96, 44, 1), output_dim=32):
    """构建用于提取梅尔图特征的CNN"""
    inp = layers.Input(shape=input_shape)
    # Block 1
    x = layers.Conv2D(CNN_FILTERS, (3, 3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # Block 2
    x = layers.Conv2D(CNN_FILTERS * 2, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)  # 使用GAP减少参数量
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Flatten()(x)
    # Dense
    x = layers.Dense(128, activation='relu')(x)
    # Output
    out = layers.Dense(output_dim, activation='relu')(x)
    return keras.Model(inputs=inp, outputs=out, name='Mel_CNN')


def build_coch_cnn(input_shape=(11, 44, 1), output_dim=32):
    """构建用于提取耳蜗图特征的CNN"""
    inp = layers.Input(shape=input_shape)
    # Block 1: 注意卷积核和池化尺寸，因为高度很小
    x = layers.Conv2D(CNN_FILTERS, (3, 3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # (11,44) -> (5, 22)
    # Block 2
    x = layers.Conv2D(CNN_FILTERS * 2, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Flatten()(x)
    # Output
    out = layers.Dense(output_dim, activation='relu')(x)
    return keras.Model(inputs=inp, outputs=out, name='Coch_CNN')


# 实例化CNN模型
mel_cnn = build_mel_cnn(MEL_SHAPE, CNN_OUTPUT_DIM)
coch_cnn = build_coch_cnn(COCH_SHAPE, CNN_OUTPUT_DIM)

# 打印模型结构查看
print("--- Mel CNN Summary ---")
mel_cnn.summary()
print("\n--- Cochleagram CNN Summary ---")
coch_cnn.summary()


class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'position': self.position,
            'd_model': self.d_model,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    """构建一个Transformer Encoder块"""
    # Multi-Head Self-Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed-Forward Network
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_mer_model():
    """构建完整的情感识别模型"""
    # 1. 定义双输入
    mel_input = layers.Input(shape=(SEQUENCE_LENGTH,) + MEL_SHAPE, name="mel_input")
    coch_input = layers.Input(shape=(SEQUENCE_LENGTH,) + COCH_SHAPE, name="coch_input")

    # 2. 并行CNN特征提取
    mel_features = layers.TimeDistributed(mel_cnn, name="time_dist_mel")(mel_input)
    coch_features = layers.TimeDistributed(coch_cnn, name="time_dist_coch")(coch_input)

    # 3. 特征融合
    fused_features = layers.Concatenate(axis=-1, name="fused_features")([mel_features, coch_features])

    # 4. 位置编码
    positional_encoder = PositionalEncoding(SEQUENCE_LENGTH, MODEL_DIM)
    x = positional_encoder(fused_features)

    # 5. Transformer Encoder 栈
    for _ in range(TRANSFORMER_LAYERS):
        x = transformer_encoder_block(x, MODEL_DIM // TRANSFORMER_HEADS, TRANSFORMER_HEADS, TRANSFORMER_UNITS,
                                      DROPOUT_RATE)

    # 6. 输出层
    output = layers.Dense(2, activation='linear', name="va_output")(x)

    # 构建模型
    model = keras.Model(inputs=[mel_input, coch_input], outputs=output, name="Music_Emotion_Transformer")
    return model


# 实例化最终模型
model = build_mer_model()
model.summary()


def _ccc_per_channel(y_true, y_pred, eps=1e-8):
    # y: [B, T, C] -> [B*T, C]，对每个通道独立做 CCC
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])  # [B*T, C]
    y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])  # [B*T, C]

    true_mean = tf.reduce_mean(y_true, axis=0)  # [C]
    pred_mean = tf.reduce_mean(y_pred, axis=0)  # [C]
    true_var = tf.math.reduce_variance(y_true, axis=0)  # [C]
    pred_var = tf.math.reduce_variance(y_pred, axis=0)  # [C]
    cov = tf.reduce_mean((y_true - true_mean) * (y_pred - pred_mean), axis=0)  # [C]

    ccc = (2.0 * cov) / (true_var + pred_var + tf.square(true_mean - pred_mean) + eps)  # [C]
    return ccc  # [C]


def ccc_loss(y_true, y_pred):
    ccc = _ccc_per_channel(y_true, y_pred)  # [C]
    return 1.0 - tf.reduce_mean(ccc)  # 标量 loss


def ccc_metric(y_true, y_pred):
    ccc = _ccc_per_channel(y_true, y_pred)  # [C]
    return tf.reduce_mean(ccc)  # 标量 metric，越大越好


# 编译模型
optimizer = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)

model.compile(
    optimizer=optimizer,
    loss=ccc_loss,
    metrics=['mean_absolute_error', ccc_metric]  # 使用单独的指标函数
)

print("\nModel compiled successfully!")

history = model.fit(
    train_dataset,
    epochs=200,
    validation_data=val_dataset,
    callbacks=[
        keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_ccc_metric', mode='max',
                                          factor=0.4, patience=4, min_lr=1e-7, verbose=1),
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    ]
)

# --- 训练后保存文件 ---

# 1. 保存最终的模型
print("正在保存最终模型到 final_model.keras ...")
model.save('final_model.keras')
print("最终模型已保存。")

# 2. 保存训练历史
print("正在保存训练历史到 training_history.npy ...")
np.save('training_history.npy', history.history)
print("训练历史已保存。")

# 使用 model.evaluate() 计算测试集上的损失和指标（参考TensorFlow官方示例）
print("\n--- 评估模型在 test_dataset 上的表现 ---")
test_results = model.evaluate(test_dataset)
# 解包结果：损失、MAE、CCC（顺序与 compile 中的 loss 和 metrics 一致）
test_loss = test_results[0]
test_mae = test_results[1]
test_ccc = test_results[2]

print(f"Test Loss (CCC Loss): {test_loss}")
print(f"Test Mean Absolute Error (MAE): {test_mae}")
print(f"Test Concordance Correlation Coefficient (CCC): {test_ccc}")

# ==============================================================================
#                 在测试集上分别计算 Valence 和 Arousal 的指标
# ==============================================================================
print("\n--- 正在为 Valence 和 Arousal 单独计算测试指标 ---")

# 步骤 1 & 2: 获取所有预测值和真实标签
print("正在从 test_dataset 生成预测...")
all_predictions = []
all_labels = []

# 遍历测试数据集，收集所有批次的预测和标签
for (mel_batch, coch_batch), labels_batch in test_dataset:
    predictions_batch = model.predict_on_batch((mel_batch, coch_batch))
    all_predictions.append(predictions_batch)
    all_labels.append(labels_batch.numpy())  # 将 TensorFlow Tensor 转换为 NumPy 数组

# 将批次列表合并成单个大的 NumPy 数组
# 初始形状: (num_batches, batch_size, sequence_length, 2)
# 合并后形状: (total_samples, sequence_length, 2)
y_pred = np.concatenate(all_predictions, axis=0)
y_true = np.concatenate(all_labels, axis=0)

# 为了计算指标，我们将时间序列展开
# 形状变为: (total_samples * sequence_length, 2)
y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
y_true_flat = y_true.reshape(-1, y_true.shape[-1])

print(f"已处理完所有测试样本，总时间步数: {y_pred_flat.shape[0]}")

# 步骤 3: 分离 Valence 和 Arousal
# y_true_v 和 y_pred_v 是 Valence 的真实值和预测值
y_true_v = y_true_flat[:, 0]
y_pred_v = y_pred_flat[:, 0]

# y_true_a 和 y_pred_a 是 Arousal 的真实值和预测值
y_true_a = y_true_flat[:, 1]
y_pred_a = y_pred_flat[:, 1]


# 步骤 4: 分别计算指标
# 我们需要一个 NumPy 版本的 CCC 函数来进行计算
def np_ccc(y_true, y_pred, eps=1e-8):
    true_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)
    true_var = np.var(y_true)
    pred_var = np.var(y_pred)

    # 计算协方差
    cov = np.mean((y_true - true_mean) * (y_pred - pred_mean))

    # 计算 CCC
    ccc = (2.0 * cov) / (true_var + pred_var + (true_mean - pred_mean) ** 2 + eps)
    return ccc


# 计算 Valence 的指标
ccc_v = np_ccc(y_true_v, y_pred_v)
mae_v = np.mean(np.abs(y_true_v - y_pred_v))

# 计算 Arousal 的指标
ccc_a = np_ccc(y_true_a, y_pred_a)
mae_a = np.mean(np.abs(y_true_a - y_pred_a))

# --- 打印最终结果 ---
print("\n--- 单独评估结果 ---")
print(f"Valence - Concordance Correlation Coefficient (CCC): {ccc_v:.4f}")
print(f"Valence - Mean Absolute Error (MAE):             {mae_v:.4f}")
print("-" * 25)
print(f"Arousal - Concordance Correlation Coefficient (CCC): {ccc_a:.4f}")
print(f"Arousal - Mean Absolute Error (MAE):             {mae_a:.4f}")
print("--------------------------")