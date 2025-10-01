# Spam Classifier - Neural Network
# References:
# - For learning the structure, backprop and forward prop of neural networks with mathematical notation: Zhang, S.,(2021) Building a neural network FROM SCRATCH [Online]. Available from: https://www.youtube.com/watch?v=w8yWXqWQYmU [Accessed 17 April 2025]
# - Backpropagation and Gradient Descent understanding: StatQuest with Josh Starmer., (2021) Neural Networks Pt. 2: Backpropagation Main Ideas [Online]. Available from: https://www.youtube.com/watch?v=IN2XmBhILt4 [Accessed 20 April 2025]
# - NumPy documentation e.g. for npz file saving: nkmk.me (n.d.) *Save and load NumPy .npz files* [Online]. Available from: https://note.nkmk.me/en/python-numpy-load-save-savez-npy-npz/ [Accessed: 21 April 2025]
# - Activation Functions Knowledge: GeeksforGeeks (n.d.) Activation functions in neural networks [Online]. Available from: https://www.geeksforgeeks.org/activation-functions-neural-networks/ [Accessed: 17 April 2025]
# Note:
# - Pre-trained weights and biases are located in 'prepped_weightbias.npz'.
# - please uncomment the classifier.train() call in the create_Classifier function for retraining.

import numpy as np
from IPython.display import HTML,Javascript, display

training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int)
print("Shape of the spam training data set:", training_spam.shape)
print(training_spam)

testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)
print("Shape of the spam testing data set:", testing_spam.shape)
print(testing_spam)

class SpamClassifier:
    #Initialise weights using He Initialisation and set bias to zero
    def __init__(self, input_data):

        self.data = np.array(input_data)
        np.random.shuffle(self.data)
        self.labels = self.data[:, 0].astype(int)   
        self.features = self.data[:, 1:].T    

        #attempt load weights
        try:
            preppedData = np.load('prepped_weightbias.npz', allow_pickle=True)
            
            self.w1 = preppedData["w1"]
            self.b1 = preppedData["b1"]
            self.w2 = preppedData["w2"] 
            self.b2 = preppedData["b2"] 
            self.w3 = preppedData["w3"]  
            self.b3 = preppedData["b3"]
        except FileNotFoundError:
            print("prepped_weightbias.npz could not be found, make sure to train the model first")      
        
        # self.w1 = np.random.randn(16, 54) * np.sqrt(2. / 54)  
        # self.b1 = np.zeros((16, 1)) 

        # self.w2 = np.random.randn(8, 16) * np.sqrt(2. / 16)   
        # self.b2 = np.zeros((8, 1)) 

        # self.w3 = np.random.randn(2, 8) * np.sqrt(2. / 8)  
        # self.b3 = np.zeros((2, 1))

    #Use of leaky ReLU helps reduce number of dead nodes; used for forward prop
    def leaky_relu(self, z):
        return np.where(z > 0, z, 0.01 * z)

    #reverse leaky ReLU function for back prop
    def dleaky_relu(self, z):
        return np.where(z > 0, 1, 0.01)

    #Sigmoid activation function for output 
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    #Create a zero array but the last index is set to One
    def one_hot (self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    #Calculate pre activation and activation values at each layer output
    def forward_prop(self, w1, b1, w2, b2, w3, b3, X):
        z1 = w1.dot(X) + b1
        a1 = self.leaky_relu(z1)
        z2 = w2.dot(a1) + b2
        a2 = self.leaky_relu(z2)
        z3 = w3.dot(a2) + b3
        a3 = self.sigmoid(z3)
        return z1, a1, z2, a2, z3, a3

    #back prop with gradient descent to reach a minimum in change of loss
    def back_prop(self, z1, a1, z2, a2, z3, a3, w3, X, Y, l2_penalty):
        m = Y.size
        one_hot_Y = self.one_hot(Y)
        dz3 = a3 - one_hot_Y
        dw3 = 1/m * dz3.dot(a2.T)
        dw3 += (l2_penalty/m) * self.w3
        db3 = 1/m * np.sum(dz3, axis=1, keepdims=True)

        #Apply reverse Leaky ReLU for back propagation
        dz2 = self.w3.T.dot(dz3) * self.dleaky_relu(z2)
        dw2 = 1/m * dz2.dot(a1.T)
        dw2 += (l2_penalty/m) * self.w2
        db2 = 1/m * np.sum(dz2, axis=1, keepdims=True)        

        dz1 = self.w2.T.dot(dz2) * self.dleaky_relu(z1)
        dw1 = 1/m * dz1.dot(X.T)
        dw1 += (l2_penalty/m) * self.w1
        db1 = 1/m * np.sum(dz1, axis=1, keepdims=True)
        
        return dw1, db1, dw2, db2, dw3, db3

    #Using alpha and change in loss with respect to weight, bias to update said parameters
    def update_params(self, dw1, db1, dw2, db2, dw3, db3, alpha):
        self.w1 -= alpha * dw1
        self.b1 -= alpha * db1
        self.w2 -= alpha * dw2
        self.b2 -= alpha * db2
        self.w3 -= alpha * dw3
        self.b3 -= alpha * db3
        # neuron_importance = np.linalg.norm(self.w1, axis=1)
        # print("Layer 1 neuron importances:", neuron_importance)

    #For each column return the index of the larger probability 0 meaning ham and 1 meaning spam
    def get_predictions(self, a3):
        return np.argmax(a3, 0)

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size

    #compute loss using cross entropy and taking L2 regularization into account
    def compute_loss(self, a3, Y, l2_penalty):
        m = Y.size
        one_hot_Y = self.one_hot(Y)
        cross_entropy = -np.sum(one_hot_Y * np.log(a3 + 1e-8)) / m
        reg_loss = (l2_penalty / (2 * m)) * (
            np.sum(np.square(self.w1)) +
            np.sum(np.square(self.w2)) +
            np.sum(np.square(self.w3))
        )
        return cross_entropy + reg_loss

    #Train the model on mini batches of the main data set for a number of iterations
    def train(self, iterations = 75, alpha=0.05, l2_penalty = 1):
        X=self.features
        Y=self.labels 
        m = X.shape[1]
        batch = 50

        for i in range(iterations):
            permutation = np.random.permutation(m)
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[permutation]

            for j in range(0, m, batch):
                X_batch = X_shuffled[:, j:j+batch]
                Y_batch = Y_shuffled[j:j+batch]             
                z1, a1, z2, a2, z3, a3 = self.forward_prop(self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, X_batch)
                dw1, db1, dw2, db2, dw3, db3 = self.back_prop(z1, a1, z2, a2, z3, a3, self.w3, X_batch, Y_batch, l2_penalty)
                self.update_params(dw1, db1, dw2, db2, dw3, db3, alpha)
                
            if i % 10 == 0:
                alpha *= 0.99
        
        np.savez('prepped_weightbias.npz', w1=self.w1, w2=self.w2, w3=self.w3, b1=self.b1, b2=self.b2, b3=self.b3)
        


    def predict(self, data):
        data = data.T
        _, _, _, _, _, a3 = self.forward_prop(self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, data)
        return self.get_predictions(a3)
    

def create_classifier():
    classifier = SpamClassifier(training_spam)

    #Uncomment if wanting to train
    #classifier.train()
    return classifier

classifier = create_classifier()
