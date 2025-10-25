import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import load_model


# 假设您的数据加载脚本名为 load_dataset.py
from load_dataset import train_dataset, val_dataset, test_dataset

# --- Dummy Data for demonstration ---
def create_dummy_dataset(num_samples, sequence_length, mel_shape, coch_shape, batch_size):
    """Creates a dummy tf.data.Dataset for testing."""
    mel_data = np.random.rand(num_samples, sequence_length, *mel_shape).astype(np.float32)
    coch_data = np.random.rand(num_samples, sequence_length, *coch_shape).astype(np.float32)
    labels = np.random.rand(num_samples, sequence_length, 2).astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(((mel_data, coch_data), labels))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# --- 模型超参数定义 ---
# 输入形状
SEQUENCE_LENGTH = 60
MEL_SHAPE = (96, 44, 1)
COCH_SHAPE = (11, 44, 1)
BATCH_SIZE = 16  # 假设的 batch size

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


print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"找到 {len(gpus)} 个 GPU，将使用: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("未找到 GPU，将使用 CPU 进行训练。")

for (mel_batch, coch_batch), labels_batch in train_dataset.take(1):
    print("--- Inspecting a single batch from the dataset ---")
    print("Mel input batch shape:", mel_batch.shape)
    print("Cochleagram input batch shape:", coch_batch.shape)
    print("Labels batch shape:", labels_batch.shape)


def build_mel_cnn(input_shape=(96, 44, 1), output_dim=32):
    """构建用于提取梅尔图特征的CNN"""
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(CNN_FILTERS, (3, 3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(CNN_FILTERS * 2, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(output_dim, activation='relu')(x)
    return keras.Model(inputs=inp, outputs=out, name='Mel_CNN')


def build_coch_cnn(input_shape=(11, 44, 1), output_dim=32):
    """构建用于提取耳蜗图特征的CNN"""
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(CNN_FILTERS, (3, 3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(CNN_FILTERS * 2, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Flatten()(x)
    out = layers.Dense(output_dim, activation='relu')(x)
    return keras.Model(inputs=inp, outputs=out, name='Coch_CNN')


mel_cnn = build_mel_cnn(MEL_SHAPE, CNN_OUTPUT_DIM)
coch_cnn = build_coch_cnn(COCH_SHAPE, CNN_OUTPUT_DIM)

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
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_mer_model():
    mel_input = layers.Input(shape=(SEQUENCE_LENGTH,) + MEL_SHAPE, name="mel_input")
    coch_input = layers.Input(shape=(SEQUENCE_LENGTH,) + COCH_SHAPE, name="coch_input")
    mel_features = layers.TimeDistributed(mel_cnn, name="time_dist_mel")(mel_input)
    coch_features = layers.TimeDistributed(coch_cnn, name="time_dist_coch")(coch_input)
    fused_features = layers.Concatenate(axis=-1, name="fused_features")([mel_features, coch_features])
    positional_encoder = PositionalEncoding(SEQUENCE_LENGTH, MODEL_DIM)
    x = positional_encoder(fused_features)
    for _ in range(TRANSFORMER_LAYERS):
        x = transformer_encoder_block(x, MODEL_DIM // TRANSFORMER_HEADS, TRANSFORMER_HEADS, TRANSFORMER_UNITS,
                                      DROPOUT_RATE)
    output = layers.Dense(2, activation='linear', name="va_output")(x)
    model = keras.Model(inputs=[mel_input, coch_input], outputs=output, name="Music_Emotion_Transformer")
    return model


model = build_mer_model()
model.summary()

# 编译模型
optimizer = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)

# ###############################################################
# # 修改部分 START
# ###############################################################

# 不再需要自定义的 ccc_loss 和 ccc_metric 函数

model.compile(
    optimizer=optimizer,
    # 1. 将 loss 更改为均方误差 (Mean Squared Error)
    loss='mean_squared_error',
    # 2. 将评估指标改为均方根误差 (Root Mean Squared Error) 和 MAE
    metrics=['mean_absolute_error', tf.keras.metrics.RootMeanSquaredError()]
)

print("\nModel compiled successfully with MSE loss and RMSE metric!")

# 3. 更新 Callbacks 以监控新的损失/指标
history = model.fit(
    train_dataset,
    epochs=200,
    validation_data=val_dataset,
    callbacks=[
        # ModelCheckpoint 监控 val_loss (验证集MSE)，并保存最低的
        keras.callbacks.ModelCheckpoint("best_model.keras", monitor='val_loss', mode='min', save_best_only=True),
        # ReduceLROnPlateau 监控 val_loss，当它不再下降时降低学习率
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min',
                                          factor=0.4, patience=4, min_lr=1e-7, verbose=1),
        # EarlyStopping 监控 val_loss，如果长时间不下降则停止训练
        keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=15, restore_best_weights=True)
    ]
)

# ###############################################################
# # 修改部分 END
# ###############################################################


# --- 训练后保存文件 ---
print("正在保存最终模型到 final_model.keras ...")
model.save('final_model_R2.keras')
print("最终模型已保存。")

print("正在保存训练历史到 training_history.npy ...")
np.save('training_history_R2.npy', history.history)
print("训练历史已保存。")