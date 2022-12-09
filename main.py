import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib


def train():
    data_dir = pathlib.Path("/home/monis/Documents/seefood/train")

    batch_size = 32

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        batch_size=batch_size)
    class_names = train_ds.class_names
    print(class_names)

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = 2

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[cp_callback]
    )

    model.evaluate(val_ds)

    model.save("./model")


def loadmodel():
    model = tf.keras.models.load_model("./model")
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    test_dir = os.path.abspath(input("img: "))

    test_image = tf.keras.utils.load_img(test_dir)
    test_image = tf.keras.utils.img_to_array(test_image)

    prediction = probability_model.predict(test_image)
    print(prediction)


loadmodel()
# train()
