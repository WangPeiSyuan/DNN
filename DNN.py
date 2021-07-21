import struct
import random
import numpy as np

'''
----- FNN configuration -----
input data: MNIST dataset, images of hand written digits from 0 to 9
output: prediction of digits 0 to 9

input layer: 784 (from 28 * 28 images)
hidden layer: 1024
output layer: 10 (prediction of digit 0~9)

activation function for hidden layer: relu
activation function for output layer: softmax

loss function: cross-entropy loss
optimizer: mini-batch gradient descent

'''


def read_MNIST_image(filename = 'train-images.idx3-ubyte'):
    file = open(filename, 'rb')
    magic_number = struct.unpack('>I', file.read(4))[0]
    image_number = struct.unpack('>I', file.read(4))[0]
    row = struct.unpack('>I', file.read(4))[0]
    col = struct.unpack('>I', file.read(4))[0]
    piexl_number = row*col
    print("Image: %d, Size: %dx%d"%(image_number,row,col))

    image_set = np.zeros((image_number,row,col))
    
    for idx in range(image_number):
        img = np.frombuffer(file.read(piexl_number), dtype=np.uint8, count=piexl_number)
        image_set[idx] = img.reshape((row,col))
        
    return image_set


def read_MNIST_label(filename = 'train-labels.idx1-ubyte'):
    file = open(filename, 'rb')
    magic_number = struct.unpack('>I', file.read(4))[0]
    label_number = struct.unpack('>I', file.read(4))[0]
    print("Label: %d"%label_number)

    label_set = np.zeros((label_number,1), dtype=np.uint8)
    
    for idx in range(label_number):
        label_set[idx] = struct.unpack('>B', file.read(1))

    return label_set


def ReLU(array): #if x<0 y=0 ; if x>0 y=x 
    value=np.multiply( np.add(array,np.abs(array)),0.5 )
    return value


def D_ReLU(array): #if x<0 y=0 ; if x>0 y=1 
    value=np.multiply( np.add(array,np.abs(array)),0.5 )
    value=np.multiply(value,np.reciprocal(array+1e-7))
    return value


def Onehot_encode(index, total):
    encode = np.zeros((1,total))
    encode[0,index] = 1
    return encode


def Softmax(array): #e^x/sum(e^x)
    array = array - np.max(array) # avoid overflow
    Output=np.exp(array)
    recipSum=np.reciprocal( np.sum(Output) )
    Output=np.multiply(Output,recipSum)
    return Output


all_image = read_MNIST_image()
all_label = read_MNIST_label()


# Train:5500 Validate:500 Test:not include
# You can select the index to do "cross-validation"
# eg: train = all_image[:5000] + all_image[10000:60000]
#     test  = all_image[5000:10000]

train_image = all_image[:55000]
train_label = all_label[:55000]

test_image = all_image[55000:]
test_label = all_label[55000:]



# initialize NN weight sample from the "standard normal" distribution
layer1W= np.random.randn(784,1024)*0.01
OutputW= np.random.randn(1024,10)*0.01
layer1b= np.random.randn(1024)*0.01
Outputb= np.random.randn(10)*0.01



# Iteration
max_iteration = 1000    # hyperparameter 
batch_size = 16       # hyperparameter 
learning_rate = 0.00001  # hyperparameter 

train_size = train_image.shape[0]
for it in range(max_iteration):
    #print("Iteration: %d"%it)

    # Get batch
    batch_index = np.random.choice(train_size, size=batch_size, replace=False)
    batchX = np.zeros((0,784))
    batchY = np.zeros((0,10))
    for idx in batch_index:
        batchX = np.vstack( (batchX, train_image[idx].flatten()) )
        batchY = np.vstack( (batchY, Onehot_encode(train_label[idx], 10)) )

    # Forward propagation
    ''' TODO: YOUR CODE '''

    
    # Backpropagation
    ''' TODO: YOUR CODE '''

    

# Validate
test_size = test_image.shape[0]

testX = test_image.reshape(-1,784)
testY = np.zeros((test_size,10))



for num in range(test_size):
    testY[num] = Onehot_encode(test_label[num], 10)

# Forward propagation
''' TODO: YOUR CODE '''



output = np.random.randn(test_size,10)
''' Random guess: you can do better than this !! '''

# Result
predict = np.argmax(output, axis=1)
answer = np.argmax(testY, axis=1)

accuarcy = np.sum(np.equal(predict, answer)/test_size)
print("Accuarcy: %f "%accuarcy)


