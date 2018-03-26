import numpy as np

def construct_feed_dict(img_inp, pkl, labels, placeholders):
	"""Construct feed dictionary."""
	coord = pkl[0]
	pool_idx = pkl[4]
	faces = pkl[5]
	# laplace = pkl[6]
	lape_idx = pkl[7]

	edges = []
	for i in range(1,4):
		adj = pkl[i][1]
		edges.append(adj[0])

	feed_dict = dict()
	feed_dict.update({placeholders['labels']: labels})
	feed_dict.update({placeholders['features']: coord})
	feed_dict.update({placeholders['img_inp']: img_inp})
	feed_dict.update({placeholders['edges'][i]: edges[i] for i in range(len(edges))})
	feed_dict.update({placeholders['faces'][i]: faces[i] for i in range(len(faces))})
	feed_dict.update({placeholders['pool_idx'][i]: pool_idx[i] for i in range(len(pool_idx))})
	feed_dict.update({placeholders['lape_idx'][i]: lape_idx[i] for i in range(len(lape_idx))})
	feed_dict.update({placeholders['support1'][i]: pkl[1][i] for i in range(len(pkl[1]))})
	feed_dict.update({placeholders['support2'][i]: pkl[2][i] for i in range(len(pkl[2]))})
	feed_dict.update({placeholders['support3'][i]: pkl[3][i] for i in range(len(pkl[3]))})
	feed_dict.update({placeholders['num_features_nonzero']: coord[1].shape})
	feed_dict.update({placeholders['dropout']: 0.})
	return feed_dict

def evaluate(label, out1, out2, out3):
	v1 = np.full([156,1], 'v')
	v2 = np.full([618,1], 'v')
	v3 = np.full([2466,1], 'v')
	obj1 = np.loadtxt('data/utils/face1.obj', dtype='|S32')
	obj2 = np.loadtxt('data/utils/face2.obj', dtype='|S32')
	obj3 = np.loadtxt('data/utils/face3.obj', dtype='|S32')

	np.savetxt('data/predict/ground.xyz', label)

	out1 = np.vstack((np.hstack((v1, out1)), obj1))
	np.savetxt('data/predict/predict1.obj', out1, fmt='%s', delimiter=' ')

	out2 = np.vstack((np.hstack((v2, out2)), obj2))
	np.savetxt('data/predict/predict2.obj', out2, fmt='%s', delimiter=' ')

	out3 = np.vstack((np.hstack((v3, out3)), obj3))
	np.savetxt('data/predict/predict3.obj', out3, fmt='%s', delimiter=' ')
	print('outputs shape', out1.shape, out2.shape, out3.shape)
