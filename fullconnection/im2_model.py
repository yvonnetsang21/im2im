import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

class Im2_Model():
    def __init__(self, input_size, hidden_size, output_size, learning_rate, reconstruct_nn_flag, train_im2_flag, model_path_nn, model_path_im2):
        """
        input_shape = (None, input_size, input_size, 2)
        #先考虑就用一整个协方差矩阵来训练
        self.data_train = tf.placeholder(tf.float32, shape=input_shape)
        conv1 = tf.layers.conv2d(
                    inputs=self.data_train,
                    filters=64,
                    kernel_size=[3, 3],
                    padding="same",
                    activation=tf.nn.relu)
        # 批量归一化层 1
        conv1 = tf.layers.batch_normalization(conv1)
        
        # 卷积层 2
        conv2 = tf.layers.conv2d(
                    inputs=conv1,
                    filters=32,
                    kernel_size=[3, 3],
                    padding="same",
                    activation=tf.nn.relu)
        # 批量归一化层 2
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)
        # ReLU激活层 2
        conv2 = tf.nn.relu(conv2)

        """
        self.data_train_ = tf.placeholder(tf.float32, shape=[None, input_size])
        self.label_im2_ = tf.placeholder(tf.float32, shape=[None, output_size])
        self.data_train = tf.transpose(self.data_train_)
        self.label_im2 = tf.transpose(self.label_im2_)

        # load nn parameter files
        if reconstruct_nn_flag == False:
            var_dict_nn = np.load(model_path_nn, allow_pickle=True).item()
        if train_im2_flag == False:
            var_dict_im2 = np.load(model_path_im2, allow_pickle=True).item()

        """
        # 单层encoder+decoder结构（注意这个地方是线性的，代码里面没有激活函数）
        # nn parameters
        if reconstruct_nn_flag == True:
            # encoder parameters
            self.W_ec = tf.Variable(initial_value=tf.random_uniform([hidden_size, input_size], minval=-0.1, maxval=0.1),
                                    trainable=True, name='W_ec')
            self.b_ec = tf.Variable(initial_value=tf.random_uniform([hidden_size, 1], minval=-0.1, maxval=0.1),
                                    trainable=True, name='b_ec')

            # decoder parameters
            self.W_dc = tf.Variable(initial_value=tf.random_uniform([output_size, hidden_size], minval=-0.1, maxval=0.1),
                                    trainable=True, name='W_dc')
            self.b_dc = tf.Variable(initial_value=tf.random_uniform([output_size, 1], minval=-0.1, maxval=0.1),
                                    trainable=True, name='b_dc')
        elif train_im2_flag == True:
            self.W_ec = tf.Variable(initial_value=var_dict_nn['W_ec:0'],
                              trainable=True, name='W_ec')
            self.b_ec = tf.Variable(initial_value=var_dict_nn['b_ec:0'],
                              trainable=True, name='b_ec')
            self.W_dc = tf.Variable(initial_value=var_dict_nn['W_dc:0'],
                               trainable=True, name='W_dc')
            self.b_dc = tf.Variable(initial_value=var_dict_nn['b_dc:0'],
                               trainable=True, name='b_dc')
        else:
            # load variable dictionary
            self.W_ec = tf.Variable(initial_value=var_dict_im2['W_ec:0'],
                                    trainable=False, name='W_ec')
            self.b_ec = tf.Variable(initial_value=var_dict_im2['b_ec:0'],
                                    trainable=False, name='b_ec')
            self.W_dc = tf.Variable(initial_value=var_dict_im2['W_dc:0'],
                                    trainable=False, name='W_dc')
            self.b_dc = tf.Variable(initial_value=var_dict_im2['b_dc:0'],
                                    trainable=False, name='b_dc')
        # output prediction
        self.h_im2 = tf.matmul(self.W_ec, self.data_train) + self.b_ec
        self.output_pred_im2 = tf.matmul(self.W_dc, self.h_im2) + self.b_dc

        # output target
        self.output_target_im2 = self.label_im2

        # loss and train
        self.error_im2 = self.output_target_im2 - self.output_pred_im2
        self.loss_im2 = tf.reduce_mean(tf.square(self.error_im2)) * (output_size)
        if train_im2_flag == True:
            self.train_op_im2 = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.loss_im2)
        """

        # 对于多层的网络，（作者为什么都要写成矩阵乘法啊。。。激活函数和dropout这些用了吗）
        # input-to-hidden, hidden-to-hidden, hidden-to-output parameters
        #self.Whi_list = []
        #self.bhi_list = []
        #self.Whh_list = []
        #self.bhh_list = []
        #self.Woh_list = []
        #self.boh_list = []
        if reconstruct_nn_flag == True:  
            self.Whi = tf.Variable(
                initial_value=tf.random_uniform([hidden_size[0], input_size], minval=-0.1, maxval=0.1),
                trainable=True, name='Whi_')
            # self.Whi_list.append(Whi_curr)
            self.bhi = tf.Variable(
                initial_value=tf.random_uniform([hidden_size[0], 1], minval=-0.1, maxval=0.1),
                trainable=True, name='bhi_')
            # self.bhi_list.append(bhi_curr)
            self.Whh1 = tf.Variable(
                initial_value=tf.random_uniform([hidden_size[1], hidden_size[0]], minval=-0.1, maxval=0.1),
                trainable=True, name='Whh1_')
            # self.Whh_list.append(Whh_curr)
            self.bhh1 = tf.Variable(
                initial_value=tf.random_uniform([hidden_size[1], 1], minval=-0.1, maxval=0.1),
                trainable=True, name='bhh1_')
            # self.bhh_list.append(bhh_curr)
            
            self.Whh2 = tf.Variable(
                initial_value=tf.random_uniform([hidden_size[2], hidden_size[1]], minval=-0.1, maxval=0.1),
                trainable=True, name='Whh2_')
            # self.Whh_list.append(Whh_curr)
            self.bhh2 = tf.Variable(
                initial_value=tf.random_uniform([hidden_size[2], 1], minval=-0.1, maxval=0.1),
                trainable=True, name='bhh2_')
            
            self.Woh = tf.Variable(
                initial_value=tf.random_uniform([output_size, hidden_size[2]], minval=-0.1, maxval=0.1),
                trainable=True, name='Woh_')
            # self.Woh_list.append(Woh_curr)
            self.boh = tf.Variable(
                initial_value=tf.random_uniform([output_size, 1], minval=-0.1, maxval=0.1),
                trainable=True, name='boh_')
            # self.boh_list.append(boh_curr)
            # self.Woo = tf.Variable(
            #     initial_value=tf.random_uniform([output_size_ss * SF_NUM, output_size_ss * SF_NUM], minval=-0.1, maxval=0.1),
            #     trainable=True, name='Woo')
            # self.boo = tf.Variable(
            #     initial_value=tf.random_uniform([output_size_ss * SF_NUM, 1], minval=-0.1,
            #                                     maxval=0.1),
            #     trainable=True, name='boo')
        
        elif train_im2_flag == True:
            self.Whi = tf.Variable(
                initial_value=var_dict_nn['Whi_' + ':0'],
                trainable=True, name='Whi_')
            # self.Whi_list.append(Whi_curr)
            self.bhi = tf.Variable(
                initial_value=var_dict_nn['bhi_' + ':0'],
                trainable=True, name='bhi_')
            # self.bhi_list.append(bhi_curr)
            self.Whh1 = tf.Variable(
                initial_value=var_dict_nn['Whh1_' + ':0'],
                trainable=True, name='Whh1_')
            # self.Whh_list.append(Whh_curr)
            self.bhh1 = tf.Variable(
                initial_value=var_dict_nn['bhh1_' + ':0'],
                trainable=True, name='bhh1_')
            # self.bhh_list.append(bhh_curr)

            self.Whh2 = tf.Variable(
                initial_value=var_dict_nn['Whh2_' + ':0'],
                trainable=True, name='Whh2_')
            # self.Whh_list.append(Whh_curr)
            self.bhh2 = tf.Variable(
                initial_value=var_dict_nn['bhh2_' + ':0'],
                trainable=True, name='bhh2_')
            
            self.Woh = tf.Variable(
                initial_value=var_dict_nn['Woh_' + ':0'],
                trainable=True, name='Woh_')
            # self.Woh_list.append(Woh_curr)
            self.boh = tf.Variable(
                initial_value=var_dict_nn['boh_' + ':0'],
                trainable=True, name='boh_')
            # self.boh_list.append(boh_curr)
            # self.Woo = tf.Variable(
            #     initial_value=var_dict_nn['Woo:0'],
            #     trainable=True, name='Woo')
            # self.boo = tf.Variable(
            #     initial_value=var_dict_nn['boo:0'],
            #     trainable=True, name='boo')
        else:
            # load variable dictionary
            self.Whi = tf.Variable(
                initial_value=var_dict_im2['Whi_' + ':0'],
                trainable=False, name='Whi_')
            #self.Whi_list.append(Whi_curr)
            self.bhi = tf.Variable(
                initial_value=var_dict_im2['bhi_' + ':0'],
                trainable=False, name='bhi_')
            #self.bhi_list.append(bhi_curr)
            self.Whh1 = tf.Variable(
                initial_value=var_dict_im2['Whh1_' + ':0'],
                trainable=False, name='Whh1_')
            #self.Whh_list.append(Whh_curr)
            self.bhh1 = tf.Variable(
                initial_value=var_dict_im2['bhh1_' + ':0'],
                trainable=False, name='bhh1_')
            #self.bhh_list.append(bhh_curr)
            
            self.Whh2 = tf.Variable(
                initial_value=var_dict_im2['Whh2_' + ':0'],
                trainable=False, name='Whh2_')
            #self.Whh_list.append(Whh_curr)
            self.bhh2 = tf.Variable(
                initial_value=var_dict_im2['bhh2_' + ':0'],
                trainable=False, name='bhh2_')
            
            self.Woh = tf.Variable(
                initial_value=var_dict_im2['Woh_' + ':0'],
                trainable=False, name='Woh_')
            #self.Woh_list.append(Woh_curr)
            self.boh = tf.Variable(
                initial_value=var_dict_im2['boh_' + ':0'],
                trainable=False, name='boh_')
            #self.boh_list.append(boh_curr)
            # self.Woo = tf.Variable(
            #     initial_value=var_dict_ss['Woo:0'],
            #     trainable=False, name='Woo')
            # self.boo = tf.Variable(
            #     initial_value=var_dict_ss['boo:0'],
            #     trainable=False, name='boo')
        
        # feed-forward
        # self.output_im2_ = []
        Whi = self.Whi
        bhi = self.bhi
        Whh1 = self.Whh1
        bhh1 = self.bhh1
        Whh2 = self.Whh2
        bhh2 = self.bhh2
        Woh = self.Woh
        boh = self.boh
        h1_ = tf.matmul(Whi, self.data_train) + bhi
        h1 = tf.tanh(h1_)
        h2_ = tf.matmul(Whh1, h1) + bhh1
        h2 = tf.tanh(h2_)
        h3_ = tf.matmul(Whh2, h2) + bhh2
        h3 = tf.tanh(h3_)
        #TODO：参考经典的encoder-decoder模型改一下激活函数、Dropout、batchnormalization等等
        #TODO：考虑给迁移学习留个口子
        self.output_im2 = tf.matmul(Woh, h3) + boh
        self.output_im2 = tf.concat(self.output_im2, axis=0)
        # self.output_ss = tf.matmul(self.Woo, tf.tanh(self.output_ss_)) + self.boo
        
        # loss and optimizer
        self.error_im2 = self.label_im2 - self.output_im2
        self.loss_im2 = tf.reduce_mean(tf.norm(tf.square(self.error_im2), ord=1))
        if train_im2_flag == True:
            self.train_op_im2 = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.loss_im2)

