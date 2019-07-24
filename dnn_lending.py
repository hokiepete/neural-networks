import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

encoder = LabelEncoder()
mean_imputer = SimpleImputer(missing_values=np.nan,strategy='mean',fill_value=None)

lending = pd.read_csv('Demo_Lending_Club.csv',header=0, sep=',')

#Drop unnecessary columns
lending.drop('Notes',axis=1,inplace=True)
lending.drop('purpose',axis=1,inplace=True)

#Remove NaNs and convert string data to numerical
emp_title = lending['emp_title']
emp_title.loc[emp_title.isnull()]='unknown'
lending['emp_title'] = encoder.fit_transform(emp_title)
del emp_title

emp_length = lending['emp_length']
emp_length.loc[emp_length=='na'] = '0'
lending['emp_length'] = encoder.fit_transform(emp_length)
del emp_length

earliest_cr_line = lending['earliest_cr_line']
earliest_cr_line.loc[earliest_cr_line.isnull()] = '0'
lending['earliest_cr_line'] = encoder.fit_transform(earliest_cr_line)
del earliest_cr_line

#Convert string data to numerical
lending['home_ownership'] = encoder.fit_transform(lending['home_ownership'])
lending['verification_status'] = encoder.fit_transform(lending['verification_status'])
lending['pymnt_plan'] = encoder.fit_transform(lending['pymnt_plan'])
lending['purpose_cat'] = encoder.fit_transform(lending['purpose_cat'])
lending['zip_code'] = encoder.fit_transform(lending['zip_code'])
lending['addr_state'] = encoder.fit_transform(lending['addr_state'])
lending['initial_list_status'] = encoder.fit_transform(lending['initial_list_status'])
lending['policy_code'] = encoder.fit_transform(lending['policy_code'])

#Replace NaNs from delinq and record data sets with (max + 1)
mths_since_last_delinq=lending['mths_since_last_delinq']
mths_since_last_delinq.loc[mths_since_last_delinq.isnull()]=mths_since_last_delinq.max()+1
del mths_since_last_delinq

mths_since_last_record = lending['mths_since_last_record']
mths_since_last_record.loc[mths_since_last_record.isnull()]=mths_since_last_record.max()+1
del mths_since_last_record

#Impute NaNs in rest of the data
mean_imputer.fit(lending)
x = mean_imputer.transform(lending)
lending = pd.DataFrame(x,columns=lending.columns)
del x

correlation_matrix = lending.corr()

y = lending['is_bad'].values
lending.drop('is_bad',axis=1,inplace=True)
x = lending.values

x_train, x_hold, y_train, y_hold = train_test_split(x,y,test_size=.2,random_state=42)



import tensorflow as tf

n_inputs = 25
n_hidden1 = 20
n_hidden2 = 16
n_hidden3 = 8
n_hidden4 = 4
n_outputs = 2

x = tf.placeholder(tf.float32, shape=(None, n_inputs), name='x')
y = tf.placeholder(tf.int64, shape=(None), name='y')

def neuron_layer(x,n_neurons,name,activation=None):
    with tf.name_scope(name):
        n_inputs = int(x.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        w = tf.Variable(init, name='weights')
        b = tf.Variable(tf.zeros([n_neurons]), name='biases')
        z = tf.matmul(x,w) + b
        if activation=='relu':
            return tf.nn.relu(z)
        else:
            return z
        
with tf.name_scope('dnn'):
    hidden1 = neuron_layer(x,n_hidden1,'hidden1',activation='relu')
    hidden2 = neuron_layer(hidden1,n_hidden2,'hidden2',activation='relu')
    hidden3 = neuron_layer(hidden2,n_hidden3,'hidden3',activation='relu')
    hidden4 = neuron_layer(hidden3,n_hidden4,'hidden4',activation='relu')
    logits = neuron_layer(hidden4,n_outputs,'outputs')
 
with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss = tf.reduce_mean(xentropy,name='loss')
    
learning_rate = 0.01
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 100
batch_size = 500

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(len(x_train[:,0]) // batch_size):
            x_batch = x_train[iteration:iteration+batch_size,:]
            y_batch = y_train[iteration:iteration+batch_size]
            sess.run(training_op,feed_dict={x:x_batch,y:y_batch})
        acc_train = accuracy.eval(feed_dict={x:x_batch,y:y_batch})
        acc_test = accuracy.eval(feed_dict={x:x_hold,y:y_hold})
        print(epoch,"Train accuracy: ", acc_train, ", Test accuracy: ",acc_test)
    save_path = saver.save(sess,"./models/dnn_lending_final.ckpt")


















