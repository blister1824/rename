import tensorflow as tf
from utils import *
from models import GCN
from DataFetcher import *

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_list', 'data/train_list.txt', 'Data list.')
flags.DEFINE_float('learning_rate', 3e-5, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 30, 'Number of epochs to train.')
flags.DEFINE_integer('hidden', 192, 'Number of units in hidden layer.')
flags.DEFINE_integer('feat_dim', 963, 'Number of units in feature layer.')
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer.')
flags.DEFINE_float('weight_decay', 5e-6, 'Weight for L2 loss.')

# Define placeholders(dict) and model
num_blocks = 3
num_supports = 2
placeholders = {
    'features': tf.placeholder(tf.float32, shape=(None, 3)),
    'img_inp': tf.placeholder(tf.float32, shape=(224, 224, 3)),
    'labels': tf.placeholder(tf.float32, shape=(None, 6)),
    'support1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support3': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'faces': [tf.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)],
    'edges': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)],
    'lape_idx': [tf.placeholder(tf.int32, shape=(None, 10)) for _ in range(num_blocks)], #for laplace term
    'pool_idx': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks-1)], #for unpooling
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}
model = GCN(placeholders, logging=True)

# Load data, initialize session
data = DataFetcher(FLAGS.data_list)
data.setDaemon(True) ####
data.start()
config=tf.ConfigProto()
#config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
#model.load(sess)

# Train graph model
train_loss = open('tmp/train_loss_record.txt', 'a')
train_loss.write('Start training, lr =  %f\n'%(FLAGS.learning_rate))
pkl = pickle.load(open('data/utils/eccv_final_version.dat', 'rb'))

train_number = data.number
for epoch in range(FLAGS.epochs):
	all_loss = np.zeros(train_number,dtype='float32') 
	for iters in range(train_number):
		# Fetch training data
		img_inp, y_train, data_id = data.fetch()
		feed_dict = construct_feed_dict(img_inp, pkl, y_train, placeholders)

		# Training step
		_, dists = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)
		all_loss[iters] = dists
		mean_loss = np.mean(all_loss[np.where(all_loss)])
		print 'Epoch %d, Iteration %d'%(epoch + 1,iters + 1)
		print 'Mean loss = %f, iter loss = %f, %d'%(mean_loss,dists,data.queue.qsize())
	# Save model
	model.save(sess)
	train_loss.write('Epoch %d, loss %f\n'%(epoch+1, mean_loss))
	train_loss.flush()
	# evaluate one model
	out1,out2,out3 = sess.run([model.output1,model.output2,model.output3], feed_dict=feed_dict)
	evaluate(y_train, out1, out2, out3)

data.shutdown()
print 'CNN-GCN Optimization Finished!'
