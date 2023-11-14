import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.w, x)
        "*** YOUR CODE HERE ***"

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        scaler = nn.DotProduct(self.w, x)
        if (nn.as_scalar(scaler)) >= 0:
            return 1
        else:
            return -1
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        batch_size = 1
        need = True
        while need:
            need = False
            for x, y in dataset.iterate_once(batch_size):
                y_scalar = nn.as_scalar(y)
                if (y_scalar != self.get_prediction(x)):
                    need = True
                    self.w.update(x, y_scalar)

        "*** YOUR CODE HERE ***"


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        self.w1 = nn.Parameter(1, 512)  # first argument should be dim (x)
        self.b1 = nn.Parameter(1, 512)
        self.w2 = nn.Parameter(512, 1)
        self.b2 = nn.Parameter(1, 1)
        self.learnRate = 0.03

        "*** YOUR CODE HERE ***"

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        h1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        predicted_y = nn.AddBias(nn.Linear(h1, self.w2), self.b2)
        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        loss = nn.SquareLoss(predicted_y, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 200
        loss = 1
        while loss >= 0.02:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                gw1, gw2, gb1, gb2 = nn.gradients(loss, [self.w1, self.w2, self.b1, self.b2])
                self.w1.update(gw1, -self.learnRate)
                self.w2.update(gw2, -self.learnRate)
                self.b1.update(gb1, -self.learnRate)
                self.b2.update(gb2, -self.learnRate)
                loss = nn.as_scalar(loss)