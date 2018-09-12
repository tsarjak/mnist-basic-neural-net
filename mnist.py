import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Read data from the mnist dataset
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)


#Number of nodes in each hidden layer
nodesl1 = 100
nodesl2 = 50
nodesl3 = 20

#Declare output class size and training batch size
classSize = 10
batchSize = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neuralNetModel(data):

    #Defining our Neural Network
    hiddenOne = {'w':tf.Variable(tf.random_normal([784, nodesl1])), 'b':tf.Variable(tf.random_normal([nodesl1]))}
    hiddenTwo = {'w':tf.Variable(tf.random_normal([nodesl1, nodesl2])), 'b':tf.Variable(tf.random_normal([nodesl2]))}
    hiddenThree = {'w':tf.Variable(tf.random_normal([nodesl2, nodesl3])), 'b':tf.Variable(tf.random_normal([nodesl3]))}
    outLayer = {'w':tf.Variable(tf.random_normal([nodesl3, classSize])), 'b':tf.Variable(tf.random_normal([classSize]))}


    #Computation for Layer one
    l1 = tf.add(tf.matmul(data,hiddenOne['w']), hiddenOne['b'])
    l1 = tf.nn.relu(l1)

    #Computation for layer two
    l2 = tf.add(tf.matmul(l1,hiddenTwo['w']), hiddenTwo['b'])
    l2 = tf.nn.relu(l2)

    #Computation for layer three
    l3 = tf.add(tf.matmul(l2,hiddenThree['w']), hiddenThree['b'])
    l3 = tf.nn.relu(l3)

    #computation for output layer
    output = tf.matmul(l3,outLayer['w']) + outLayer['b']


    #Return final prediction vector
    return output


def trainModel(x):

    #Get neural net predicted output for the model
    predictedOutput = neuralNetModel(x)

    #Calculate cost function of the model (Softmax)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=predictedOutput, labels=y) )

    #Declare optimizing function (Adam Optimizer)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    #Declare number of epochs
    numEpochs = 100


    #Running the session
    with tf.Session() as sess:
        
        #Initialize global variables
        sess.run(tf.global_variables_initializer())

        #Running epochs
        for epoch in range(numEpochs):

            epochLoss = 0

            for i in range(int(mnist.train.num_examples/batchSize)):
                epoch_x, epoch_y = mnist.train.next_batch(batchSize)
                i, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epochLoss += c

            print('Epoch', epoch + 1, 'completed out of',numEpochs,'loss:',epochLoss)

        correct = tf.equal(tf.argmax(predictedOutput, 1), tf.argmax(y, 1))


        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


trainModel(x)