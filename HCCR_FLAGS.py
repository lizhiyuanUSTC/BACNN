import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 450000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('log_steps', 20,
                            """Number of batches to logging.""")
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 10000,
                            """Number of batches to saving model.""")
tf.app.flags.DEFINE_float('LEARNING_RATE_DECAY_FACTOR', 0.1,
                            """Learning rate devay factor.""")
tf.app.flags.DEFINE_float('INITIAL_LEARNING_RATE', 0.1,
                            """Initial learning rate.""")
tf.app.flags.DEFINE_integer('decay_steps', 200000,
                            """Number of batches to learning rate decay.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('batch_size', 256,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('eval_batch_size', 999,
                            """Number of images to process in a batch when evaluation.""")
tf.app.flags.DEFINE_string('train_data', '/media/ai/DL_DATA/HCCR/THRESH_OTSU/train.tfrecords/',
                           """Path to the HCCR training data directory.""")
tf.app.flags.DEFINE_string('competition_data', '/media/ai/DL_DATA/HCCR/THRESH_OTSU/competition.tfrecords/',
                           """Path to the HCCR competing data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('IMAGE_SIZE', 64,
                            """Image Size.""")
tf.app.flags.DEFINE_integer('NUM_CLASSES', 3755,
                            """NUM_CLASSES.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 16,
                            """Number of threads to reading data.""")
tf.app.flags.DEFINE_integer('embedding_size', 512,
                            """Dimensionality of the embedding.""")
tf.app.flags.DEFINE_float('weight_decay', 0.0,
                            """Initial learning rate.""")

