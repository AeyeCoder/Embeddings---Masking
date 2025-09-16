import tensorflow_datasets as tfds
import tensorflow as tf

raw_train,raw_val,raw_test=tfds.load(
    name="imdb_reviews",
    split=["train","test[:50%]","test[50%:]"],
    as_supervised=True
)
tf.random.set_seed(42)
train=raw_train.shuffle(buffer_size=30000,seed=42).batch(32).prefetch(1)
val=raw_val.batch(32).prefetch(1)
test=raw_test.batch(32).prefetch(1)
vocab_size=1200
text_vec_layer=tf.keras.layers.TextVectorization(max_tokens=vocab_size)
text_vec_layer.adapt(train.map(lambda reviews,labels:reviews))
embed_size=128

model=tf.keras.Sequential([
    text_vec_layer,
    tf.keras.layers.Embedding(vocab_size,embed_size,mask_zero=True),
    tf.keras.layers.GRU(128,return_sequences=False),
    tf.keras.layers.Dense(1,activation="sigmoid")
])

model.compile(loss="binary_crossentropy",optimizer="nadam",metrics=["Accuracy"])
model.fit(train,validation_data=val,epochs=4)
