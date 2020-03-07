"""Train DNN on census income dataset."""

import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import pandas as pd

raw_file = './child_app_feat'
f_map = './fn_fid_map.txt'

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


def hiveData2ndarray(input_file, num_feat, m):
    reader = open(input_file, 'rb')
    train_data_array = []
    target_data_array = []
    #     m = {}
    index = 0
    for line in reader:
        train_data = []

        line = line.strip().split(b'\t')
        label = line[1]
        # label = [1, 0] if label == 0 else [0, 1]
        target_data_array.append(label)
        line = line[2].strip().split(b',')

        train_data = [0] * num_feat

        for l in line:
            l = l.split(b':')
            train_data[int(m[int(l[0])])] = int(l[1])
        train_data_array.append(train_data[:num_feat])
    # print(input_file, "m:", len(m))
    return train_data_array, target_data_array


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    # print("dataset:*", dataset)
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    m = {}
    with open(f_map, 'r') as f:
        for line in f:
            line = line.split(' ')
            m[int(line[0])] = line[1]
    # print(m)
    keys = ['f_' + str(i) for i in range(1973)]
    f_data, target_data = hiveData2ndarray(raw_file, 1973, m)
    # print("f_data type:", f_data)
    train_x, test_x, train_y, test_y = train_test_split(f_data, target_data, test_size=0.3)
    # (train_x, train_y), (test_x, test_y) = iris_data.load_data()
    train_x = pd.DataFrame(train_x, columns=keys)
    test_x = pd.DataFrame(test_x, columns=keys)
    train_y = pd.Series(train_y)
    test_y = pd.Series(test_y)
    # print("train_x type:", type(train_x[:]))
    # print("train_y type:", type(train_y))

    # Feature columns describe how to use the input.
    my_feature_columns = []

    # print(keys)
    for key in keys:
        # print("key:", key)
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # print("my_feature_columns:", my_feature_columns)

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=2
    )

    # Train the Model.
    # print("asdsdas:", type(train_x))
    # print("asdsdas:", np.shape(train_x))
    # print("asdsdas:", (train_x[:2]))
    # print("args.train_steps:", args.train_steps)
    print("Start to train......")
    classifier.train(
        input_fn=lambda:train_input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps)
    print("Train success!")
    # Evaluate the model_v2.

    print("Start to eval......")
    eval_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn(test_x, test_y, args.batch_size))
    print("Eval success!")
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model_v2
    # expected = ['Setosa', 'Versicolor', 'Virginica']
    # predict_x = {
    #     'SepalLength': [5.1, 5.9, 6.9],
    #     'SepalWidth': [3.3, 3.0, 3.1],
    #     'PetalLength': [1.7, 4.2, 5.4],
    #     'PetalWidth': [0.5, 1.5, 2.1],
    # }
    #
    #
    # predictions = classifier.predict(
    #     input_fn=lambda: eval_input_fn(predict_x,
    #                                             labels=None,
    #                                             batch_size=args.batch_size))
    #
    # template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
    #
    # for pred_dict, expec in zip(predictions, expected):
    #     class_id = pred_dict['class_ids'][0]
    #     probability = pred_dict['probabilities'][class_id]
    #
    #     print(template.format(iris_data.SPECIES[class_id],
    #                           100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)





#
# def define_census_flags():
#   wide_deep_run_loop.define_wide_deep_flags()
#   flags.adopt_module_key_flags(wide_deep_run_loop)
#   flags_core.set_defaults(data_dir='/tmp/census_data',
#                           model_dir='/tmp/census_model',
#                           train_epochs=40,
#                           epochs_between_evals=2,
#                           inter_op_parallelism_threads=0,
#                           intra_op_parallelism_threads=0,
#                           batch_size=40)
#
#
# def build_estimator(model_dir, model_type, model_column_fn, inter_op, intra_op):
#   """Build an estimator appropriate for the given model_v2 type."""
#   wide_columns, deep_columns = model_column_fn()
#   hidden_units = [100, 75, 50, 25]
#
#   # Create a tf.estimator.RunConfig to ensure the model_v2 is run on CPU, which
#   # trains faster than GPU for this model_v2.
#   run_config = tf.estimator.RunConfig().replace(
#       session_config=tf.ConfigProto(device_count={'GPU': 0},
#                                     inter_op_parallelism_threads=inter_op,
#                                     intra_op_parallelism_threads=intra_op))
#
#   if model_type == 'wide':
#     return tf.estimator.LinearClassifier(
#         model_dir=model_dir,
#         feature_columns=wide_columns,
#         config=run_config)
#   elif model_type == 'deep':
#     return tf.estimator.DNNClassifier(
#         model_dir=model_dir,
#         feature_columns=deep_columns,
#         hidden_units=hidden_units,
#         config=run_config)
#   else:
#     return tf.estimator.DNNLinearCombinedClassifier(
#         model_dir=model_dir,
#         linear_feature_columns=wide_columns,
#         dnn_feature_columns=deep_columns,
#         dnn_hidden_units=hidden_units,
#         config=run_config)
#
#
# def run_census(flags_obj):
#   """Construct all necessary functions and call run_loop.
#   Args:
#     flags_obj: Object containing user specified flags.
#   """
#   if flags_obj.download_if_missing:
#     census_dataset.download(flags_obj.data_dir)
#
#   train_file = os.path.join(flags_obj.data_dir, census_dataset.TRAINING_FILE)
#   test_file = os.path.join(flags_obj.data_dir, census_dataset.EVAL_FILE)
#
#   # Train and evaluate the model_v2 every `flags.epochs_between_evals` epochs.
#   def train_input_fn():
#     return census_dataset.input_fn(
#         train_file, flags_obj.epochs_between_evals, True, flags_obj.batch_size)
#
#   def eval_input_fn():
#     return census_dataset.input_fn(test_file, 1, False, flags_obj.batch_size)
#
#   tensors_to_log = {
#       'average_loss': '{loss_prefix}head/truediv',
#       'loss': '{loss_prefix}head/weighted_loss/Sum'
#   }
#
#   wide_deep_run_loop.run_loop(
#       name="Census Income", train_input_fn=train_input_fn,
#       eval_input_fn=eval_input_fn,
#       model_column_fn=census_dataset.build_model_columns,
#       build_estimator_fn=build_estimator,
#       flags_obj=flags_obj,
#       tensors_to_log=tensors_to_log,
#       early_stop=True)
#
#
# def main(_):
#   with logger.benchmark_context(flags.FLAGS):
#     run_census(flags.FLAGS)
#
#
# if __name__ == '__main__':
#   tf.logging.set_verbosity(tf.logging.INFO)
#   define_census_flags()
#   absl_app.run(main)