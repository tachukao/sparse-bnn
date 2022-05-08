from typing import Dict

from absl import app, flags
import sonnet as snt
import tensorflow as tf
import tensorflow_datasets as tfds
import networks as netlib

import os
import uuid

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

FLAGS = flags.FLAGS

flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
flags.DEFINE_float("anneal_rate", None, "Anneal rate.")
flags.DEFINE_integer("train_batch_size", 128, "Training batch size.")
flags.DEFINE_integer("test_batch_size", 1000, "Testing batch size.")
flags.DEFINE_integer("num_epochs", 500, "Number of training epochs.")
flags.DEFINE_string("save_dir", "/scratches/cblgpu03/tck29/sbnn", "Results directory.")
flags.DEFINE_string("prior", "normal", "Prior placed on weights.")


def mnist(split: str, batch_size: int) -> tf.data.Dataset:
    """Returns a tf.data.Dataset with MNIST image/label pairs."""

    def preprocess_dataset(images, labels):
        # Mnist images are int8 [0, 255], we cast and rescale to float32 [-1, 1].
        images = ((tf.cast(images, tf.float32) / 255.0) - 0.5) * 2.0
        return images, labels

    dataset, info = tfds.load(
        name="mnist",
        split=split,
        shuffle_files=split == "train",
        as_supervised=True,
        with_info=True,
    )
    dataset = dataset.map(preprocess_dataset)
    dataset = dataset.shuffle(buffer_size=4 * batch_size)
    dataset = dataset.batch(batch_size)
    # Cache the result of the data pipeline to avoid recomputation. The pipeline
    # is only ~100MB so this should not be a significant cost and will afford a
    # decent speedup.
    dataset = dataset.cache()
    # Prefetching batches onto the GPU will help avoid us being too input bound.
    # We allow tf.data to determine how much to prefetch since this will vary
    # between GPUs.
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, info


def anneal_fn(step):
    if FLAGS.anneal_rate is None:
        return 1.0
    step = tf.cast(step, tf.float32)
    return 1 - tf.math.exp(-step * FLAGS.anneal_rate)


def neg_log_prob(labels, logits):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)


def train_step(
    model: snt.Module,
    optimizer: snt.Optimizer,
    images: tf.Tensor,
    labels: tf.Tensor,
) -> tf.Tensor:
    """Runs a single training step of the model on the given input."""
    with tf.GradientTape() as tape:
        logits = model(images)
        nlp = tf.reduce_mean(neg_log_prob(labels, logits))
        # TODO: do not hard-code training data size this
        data_size = 60000.0
        kl = model.kl / data_size
        anneal = anneal_fn(optimizer.step)
        loss = nlp + anneal * kl
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply(gradients, variables)

    return {
        "loss": loss,
        "nlp": nlp,
        "kl": kl,
        "anneal": anneal,
        "scale": model.scale,
    }


@tf.function
def train_epoch(
    model: snt.Module,
    optimizer: snt.Optimizer,
    dataset: tf.data.Dataset,
) -> tf.Tensor:
    log = {
        "loss": 0.0,
        "nlp": 0.0,
        "kl": 0.0,
        "anneal": 0.0,
        "scale": 0.0,
    }
    for images, labels in dataset:
        log = train_step(model, optimizer, images, labels)
    return log


@tf.function
def test_accuracy(
    model: snt.Module,
    dataset: tf.data.Dataset,
) -> Dict[str, tf.Tensor]:
    """Computes accuracy on the test set."""
    correct, total = 0, 0
    for images, labels in dataset:
        preds = tf.argmax(model.predict(images), axis=1)
        correct += tf.math.count_nonzero(tf.equal(preds, labels), dtype=tf.int32)
        total += tf.shape(labels)[0]
    accuracy = (correct / tf.cast(total, tf.int32)) * 100.0
    return {"accuracy": accuracy, "incorrect": total - correct, "total": total}


def main(unused_argv):
    del unused_argv

    model = netlib.BNN(prior=FLAGS.prior)
    optimizer = snt.optimizers.Adam(FLAGS.learning_rate)

    train_data, train_info = mnist("train", batch_size=FLAGS.train_batch_size)
    test_data, test_info = mnist("test", batch_size=FLAGS.test_batch_size)

    uid = str(uuid.uuid4())
    save_dir = os.path.join(FLAGS.save_dir, FLAGS.prior, uid)
    os.makedirs(save_dir, exist_ok=True)
    saved_model_dir = os.path.join(save_dir, "saved_model")
    os.makedirs(saved_model_dir, exist_ok=True)

    for epoch in range(FLAGS.num_epochs):
        train_metrics = train_epoch(model, optimizer, train_data)
        test_metrics = test_accuracy(model, test_data)
        if epoch % 10 == 0:
            to_save = snt.Module()
            to_save.all_variables = list(model.variables)
            tf.saved_model.save(to_save, saved_model_dir)
        print(
            "[Epoch %d] loss: %.05f, nlp: %.05f, kl: %.05f, anneal: %.05f, scale: %.05f, test acc: %.02f%% (%d/%d wrong)"
            % (
                epoch,
                train_metrics["loss"],
                train_metrics["nlp"],
                train_metrics["kl"],
                train_metrics["anneal"],
                train_metrics["scale"],
                test_metrics["accuracy"],
                test_metrics["incorrect"],
                test_metrics["total"],
            )
        )


if __name__ == "__main__":
    app.run(main)
