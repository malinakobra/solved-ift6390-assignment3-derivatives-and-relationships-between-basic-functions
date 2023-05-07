Download Link: https://assignmentchef.com/product/solved-ift6390-assignment3-derivatives-and-relationships-between-basic-functions
<br>
<h2>1 THEORETICAL PART a  : derivatives and relationships between basic functions</h2>

Given

—  logistic sigmoid  sigmoid(.

—  hyperbolic tangent.

—  softplus  softplus(<em>x</em>) = ln(1 + exp(<em>x</em>))

—  function sign which returns +1 if its argument is positive, -1 if negative and 0 if 0.

— <strong>1</strong><em><sub>S</sub></em>(<em>x</em>) is the indicator function which returns 1 if <em>x </em>∈ <em>S </em>(or <em>x </em>respects condition <em>S</em>), otherwise returns 0.

—  rectifier  function which keeps only the positive part of its argument : rect(<em>x</em>) returns <em>x </em>if <em>x </em>≥ 0 and returns 0 if <em>x &lt; </em>0. It is also named RELU (rectified linear unit) : rect(<em>x</em>) = RELU(<em>x</em>) = [<em>x</em>]<sub>+ </sub>= max(0<em>,x</em>) = <strong>1</strong>{<em>x&gt;</em>0}(<em>x</em>)

<ol>

 <li>Show that sigmoid(</li>

 <li>Show that lnsigmoid(<em>x</em>) = −softplus(−<em>x</em>)</li>

 <li>Show that the derivative of the sigmoid is : sigmoid<sup>0</sup>(<em>x</em>) = <sup>dsigmoid</sup><sub>d<em>x </em></sub>(<em>x</em>) = sigmoid(<em>x</em>)(1 − sigmoid(<em>x</em>))</li>

 <li>Show that the tanh derivative is : tanh<sup>0</sup>(<em>x</em>) = 1 − tanh<sup>2</sup>(<em>x</em>)</li>

 <li>Write the sign function using only indicator functions : sign(<em>x</em>) = <em>…</em></li>

 <li>Write the derivative of the absolute function abs(<em>x</em>) = |<em>x</em>|. Note : its derivative at 0 is not defined, but your function abs<sup>0 </sup>can return 0 at 0. Note 2 : use the sign function : abs<sup>0</sup>(<em>x</em>) = <em>…</em></li>

 <li>Write the derivative of the function rect. Note : its derivative at 0 is undefined, but your function can return 0 at 0. Note2 : use the indicator function. rect<sup>0</sup>(<em>x</em>) = <em>…</em></li>

 <li>Let the squared <em>L</em><sub>2 </sub>norm of a vector be :. Write the vector of the gradient :</li>

 <li>Let the norm <em>L</em><sub>1 </sub>of a vector be : k<strong>x</strong>k<sub>1 </sub>= <sup>P</sup><em><sub>i </sub></em>|<strong>x</strong><em><sub>i</sub></em>|. Write the vector of the gradient :</li>

</ol>

<h2>2 THEORETICAL PART b  : Gradient computation for parameters optimization in a neural net for multiclass classification</h2>

Let <em>D<sub>n </sub></em>= {(<strong>x</strong><sup>(1)</sup><em>,y</em><sup>(1)</sup>)<em>,…,</em>(<strong>x</strong><sup>(<em>n</em>)</sup><em>,y</em><sup>(<em>n</em>)</sup>)} be the dataset with <em>x</em><sup>(<em>i</em>) </sup>∈ R<em><sup>d </sup></em>and <em>y</em><sup>(<em>i</em>) </sup>∈ {1<em>,…,m</em>} indicating the class within <em>m </em>classes. For vectors and matrices in the following equations, vectors are by default considered to be column vectors.

Consider a neural net of the type <em>Multilayer perceptron </em>(MLP) with only one hidden layer (meaning 3 layers total if we count the input and output layers). The hidden layer is made of <em>d<sub>h </sub></em>neurons fully connected to the input layer. We shall consider a non linearity of type <strong>rectifier </strong>(Rectified Linear Unit or <strong>RELU</strong>) for the hidden layer. The output layer is made of <em>m </em>neurons that are fully connected to the hidden layer. They are equipped with a <strong>softmax </strong>non linearity. The output of the <em>j</em><sup>th </sup>neuron of the output layer gives a score for the class <em>j </em>which is interpreted as the probability of <em>x </em>being of class <em>j</em>.

It is highly recommended that you draw the neural net as it helps understanding all the steps.

<ol>

 <li>Let <strong>W</strong><sup>(1) </sup>a <em>d<sub>h </sub></em>× <em>d </em>matrix of weights and <strong>b</strong><sup>(1) </sup>the bias vector be the connections between the input layer and the hidden layer. What is the dimension of <strong>b</strong><sup>(1) </sup>? Give the formula of the preactivation vector (before the non linearity) of the neurons of the hidden layer <strong>h</strong><em><sup>a </sup></em>given <strong>x </strong>as input, first in a matrix form (<em>h<sup>a </sup></em>= <em>…</em>), and then details on how to compute one element <strong>h</strong><em><sup>a</sup><sub>j </sub></em>= <em>…</em>. Write the output vector of the hidden layer <strong>h</strong><em><sup>s </sup></em>with respect to <strong>h</strong><em><sup>a</sup></em>.</li>

 <li>Let <strong>W</strong><sup>(2) </sup>a weight matrix and <strong>b</strong><sup>(2)</sup>a bias vector be the connections between the hidden layer and the output layer. What are the dimensions of <strong>W</strong><sup>(2) </sup>and <strong>b</strong><sup>(2) </sup>? Give the formula of the activation function of the neurons of the output layer <strong>o</strong><em><sup>a </sup></em>with respect to their input <strong>h</strong><em><sup>s </sup></em>in a matrix form and then write in a detailed form for <strong>o</strong><em><sup>a</sup><sub>k</sub></em>.</li>

 <li>The output of the neurons at the output layer is given by :</li>

</ol>

<strong>o</strong><em><sup>s        </sup></em>=      softmax(<strong>o</strong><em><sup>a</sup></em>)

Give the precise equation for <strong>o</strong><em><sup>s</sup><sub>k </sub></em>using the softmax (formula with the exp). <em>Show </em>that the <strong>o</strong><em><sup>s</sup><sub>k </sub></em>are positive and sum to 1. Why is this important?

<ol start="4">

 <li>The neural net computes, for an input vector <strong>x</strong>, a vector of probability scores <strong>o</strong><em><sup>s</sup></em>(<strong>x</strong>). The probability, computed by a neural net, that an observation <strong>x </strong>belong to class <em>y </em>is given by the <em>y</em><sup>th </sup>output). This suggests a loss function such as :</li>

</ol>

Find the equation of <em>L </em>as a function of the vector <strong>o</strong><em><sup>a</sup></em>. It is easily achievable with the correct substitution using the equation of the previous question.

<ol start="5">

 <li>The training of the neural net will consist of finding parameters that minimize the empirical risk <em>R</em>ˆ associated with this loss function. What is <em>R</em>ˆ ? What is precisely the set <em>θ </em>of parameters of the network? How many scalar parameters <em>n<sub>θ </sub></em>are there? Write down the optimization problem of training the network in order to find the optimal values for these parameters.</li>

 <li>To find a solution to this optimization problem, we will use gradient descent. What is the (batch) gradient descent equation for this problem?</li>

 <li>We can compute the vector of the gradient of the empirical risk <em>R</em>ˆ with respect to the parameters set <em>θ </em>this way</li>

</ol>

This hints that we only need to know how to compute the gradient of the loss <em>L </em>with an example(<strong>x</strong><em>,y</em>) with respect to the parameters, defined as followed :

We shall use <strong>gradient backpropagation</strong>, starting with loss <em>L </em>and going to the output layer <strong>o </strong>then down the hidden layer <strong>h </strong>then finaly at the input layer <strong>x</strong>.

<strong>Show that</strong>

onehot<em><sub>m</sub></em>(<em>y</em>)

Note : Start from the expression of <em>L </em>as a function of <strong>o</strong><em><sup>a </sup></em>that you previously found. Start by computing (using the start of the expression of the logarithm derivate). Do the same thing for .

<ol start="8">

 <li>What is the numpy equivalent expression (it can fit in 2 operations)?</li>

</ol>

grad oa = <em>…</em>

<em>…</em>

<strong>IMPORTANT : </strong>From now on when we ask to ”compute” the gradients or partial derivates, you only need to write them as function of previously computed derivates (<strong>do not substitute the whole expressions already computed in the previous questions!</strong>)

<ol start="9">

 <li>Compute the gradients with respect to parameters <strong>W</strong><sup>(2) </sup>and <strong>b</strong><sup>(2) </sup>of the</li>

</ol>

(2)                   (2)                   <em><sup>a </sup></em>the result output layer. Since <em>L </em>depends on <strong>W</strong><em><sub>kj </sub></em>and <strong>b</strong><em><sub>k </sub></em>only through <strong>o</strong><em><sub>k </sub></em>of the chain rule is : and

<ol start="10">

 <li>Write down the gradient of the last question in matrix form and define the dimensions of all matrix or vectors involved.</li>

</ol>

(What are the dimensions?)

Take time to understand why the above equalities are the same as the equations of the last question.

Give the numpy form : grad b2 = <em>…</em>

grad W2 = <em>…</em>

<ol start="11">

 <li>What is the partial derivate of the loss <em>L </em>with respect to the output of the neurons at the hidden layer? Since <em>L </em>depends on <strong>h</strong><em><sup>s</sup><sub>j </sub></em>only through the activations of the output neurons <strong>o</strong><em><sup>a </sup></em>the chain rule yields :</li>

 <li>Write down the gradient of the last question in matrix form and define the dimensions of all matrix or vectors involved.</li>

</ol>

(What are the dimensions?)

Take time to understand why the above equalities are the same as the equations of the last question.

Give the numpy form : grad hs = <em>…</em>

<ol start="13">

 <li>What is the partial derivate of the loss <em>L </em>with respect to the activation of the neurons at the hidden layer? Since <em>L </em>depends on the activation <strong>h</strong><em><sup>a</sup><sub>j </sub></em>only through <strong>h</strong><em><sup>s</sup><sub>j </sub></em>of this neuron, the chain rule gives :</li>

</ol>

Note <strong>h</strong> = rect( ) : the rectifier function is applied elementwise. Start by writing the derivate of the rectifier function = rect<sup>0</sup>(<em>z</em>) = <em>…</em>.

<ol start="14">

 <li>Write down the gradient of the last question in matrix form and define the dimensions of all matrix or vectors involved. Give the numpy form.</li>

 <li>What is the gradient with respect to the parameters <strong>W</strong><sup>(1) </sup>and <strong>b</strong><sup>(1) </sup>of the hidden layer?</li>

</ol>

<h3>Note : same logic as a previous question</h3>

<ol start="16">

 <li>Write down the gradient of the last question in matrix form and define the dimensions of all matrix or vectors involved. Give the numpy form. <strong>Note : same logic as a previous question</strong></li>

 <li>What are the partial derivates of the loss <em>L </em>with respect to <strong>x</strong>? <strong>Note : same logic as a previous question</strong></li>

 <li>We will now consider a <strong>regularized </strong>emprical risk : <em>R</em>˜ = <em>R</em>ˆ + L(<em>θ</em>), where <em>θ </em>is the vector of all the parameters in the network and L(<em>θ</em>) describes a scalar penalty as a function of the parameters <em>θ</em>. The penalty is given importance according to a prior preferences for the values of <em>θ</em>. The <em>L</em><sub>2 </sub>(quadratic) regularization that penalizes the square norm (norm <em>L</em><sub>2</sub>) of the weights (but not the biases) is more standard, is used in ridge regression and is sometimes called ”weight-decay”. Here we shall consider a double regularization <em>L</em><sub>2 </sub>and <em>L</em><sub>1 </sub>which is sometimes named “elastic net” and we will use different <strong>hyperparameters </strong>(positive scalars <em>λ</em><sub>11</sub><em>,λ</em><sub>12</sub><em>,λ</em><sub>21</sub><em>,λ</em><sub>22</sub>) to control the effect of the regularization at each layer</li>

</ol>

<table width="413">

 <tbody>

  <tr>

   <td width="39">L(<em>θ</em>)</td>

   <td width="24">=</td>

   <td width="350">L(<strong>W</strong>(1)<em>,</em><strong>b</strong>(1)<em>,</em><strong>W</strong>(2)<em>,</em><strong>b</strong>(2))</td>

  </tr>

  <tr>

   <td width="39"> </td>

   <td width="24">=</td>

   <td width="350"></td>

  </tr>

  <tr>

   <td width="39"> </td>

   <td width="24">=</td>

   <td width="350"></td>

  </tr>

 </tbody>

</table>

<em>i,j                                                         ij                                                             i,j</em>

                       

+<em>λ</em>22 X(<strong>W</strong><em>ij</em>(2))2

<em>ij</em>

We will in fact minimize the regularized risk <em>R</em>˜ instead of <em>R</em>ˆ. How does this change the gradient with respect to the different parameters?

<h2>3 PRACTICAL PART  : Neural network implementation and experiments</h2>

We ask you to implement a neural network where you compute the gradients using the formulas derived in the previous part (including elastic net type regularization). You must not use an existing neural network library, but you must use the derivation of part 2 (with corresponding variable names, etc). Note that you can reuse the general learning algorithm structure that we used in the demos, as well as the functions used to plot the decision functions.

<strong>Useful details on implementation </strong>:

— <strong>Numerically stable softmax</strong>. You will need to compute a numerically stable softmax. Refer to lecture notes for a proper way of computing a numerically stable softmax. Start by writing the expression for a single vector, then adapt it for a mini-batch of examples stored in a matrix.

— <strong>Parameter initialization. </strong>As you know, it is necessary to randomly initialize the parameters of your neural network (trying to avoid symmetry and saturating neurons, and ideally so that the pre-activation lies in the bending region of the activation function so that the overall networks acts as a non linear function). We suggest that you sample the weights of a layer from a uniform distribution in , where <em>n<sub>c </sub></em>is the number of inputs for <strong>this layer </strong>(changing from one layer to the other). <em>Biases </em>can be initialized at 0. Justify any other initialization method.

— <strong>fprop </strong>and <strong>bprop</strong>. We suggest you implement methods fprop and bprop. fprop will compute the forward progpagation i.e. step by step computation from the input to the output and the cost, of the activations of each layer. bprop will use the computed activations by fprop and does the backpropagation of the gradients from the cost to the input following precisely the steps derived in part 2.

— <strong>Finite difference gradient check. </strong>We can estimate the gradient numerically using the finite difference method. You will implement this estimate as a tool to check your gradient computation. To do so, calculate the value of the loss function for the current parameter values (for a single example or a mini batch). Then for each scalar parameter <em>θ<sub>k</sub></em>, change the parameter value by adding a small perturbation and calculate the new value of the loss (same example or minibatch), then set the value of the parameter back to its original value. The partial derivative with respect to this parameter is estimated by dividing the change in the loss function by <em>ε</em>. The ratio of your gradient computed by backpropagation and your estimate using finite difference should be between 0<em>.</em>99 and 1<em>.</em>01.

— <strong>Size of the mini batches</strong>. We ask that your computation and gradient descent is done in minibatches (as opposed to the whole training set) with adjustable size using a hyperparameter <em>K</em>. In the minibatch case, we do not manipulate a single input vector, but rather a batch of input vectors grouped in a matrix (that will give a matrix representation at each layer, and for the input). In the case where the size is one, we obtain an equivalent to the stochastic gradient. Given that numpy is efficient on matrix operations, it is more efficient to perform computations on a whole minibatch. It will greatly impact the execution time.

<strong>Experiments : </strong>We will use the two circles dataset and the task of classifying pieces of clothes from fashion MNIST (see links on course website).

<ol>

 <li>As a beginning, start with an implementation that computes the gradients for <strong>a single </strong>example, and check that the gradient is correct using the finite difference method described above.</li>

 <li>Display the gradients for both methods (direct computation and finite difference) for a small network (e.g. <em>d </em>= 2 and <em>d<sub>h </sub></em>= 2) with random weights and for a single example.</li>

 <li>Add a hyperparameter for the minibatch size <em>K </em>to allow compute the gradients on a minibatch of <em>K </em>examples (in a matrix), by <strong>looping </strong>over the <em>K </em>examples (this is a small addition to your previous code).</li>

 <li>Display the gradients for both methods (direct computation and finite difference) for a small network (e.g. <em>d </em>= 2 and <em>d<sub>h </sub></em>= 2) with random weights and for a minibatch with 10 examples (you can use examples from both classes from the two circles dataset).</li>

 <li>Train your neural network using gradient descent on the two circles dataset. Plot the decision regions for several different values of the hyperparameters (weight decay, number of hidden units, early stopping) so as to illustrate their effect on the capacity of the model.</li>

 <li>As a second step, copy your existing implementation to modify it to a new implementation that will use matrix calculus (instead of a loop) on batches of size <em>K </em>to improve efficiency. <strong>Take the matrix expressions in numpy derived in the first part, and adapt them for a minibatch of size </strong><em>K</em><strong>. Show in your report what you have modified (describe the former and new expressions with the shapes of each matrices).</strong></li>

 <li>Compare both implementations (with a loop and with matrix calculus) to check that they both give the same values for the gradients on the parameters, first for <em>K </em>= 1, then for <em>K </em>= 10. Display the gradients for both methods.</li>

 <li>Time how long takes an epoch on fashion MNIST (1 epoch = 1 full traversal through the whole training set) for <em>K </em>= 100 for both versions (loop over a minibatch and matrix calculus).</li>

 <li>Adapt your code to compute the error (proportion of misclassified examples) on the training set as well as the total loss on the training set during each epoch of the training procedure, and at the end of each epoch, it computes the error and average loss on the validation set and the test set. Display the 6 corresponding figures (error and average loss on train/valid/test), and write them in a log file.</li>

 <li>Train your network on the fashion MNIST dataset. Plot the training/valid/test curves (error and loss as a function of the epoch number, corresponding</li>

</ol>

to what you wrote in a file in the last question). Add to your report the curves obtained using your best hyperparameters, i.e. for which you obtained your best error on the validation set. We suggest 2 plots : the first one will plot the error rate (train/valid/test with different colors, show which color in a legend) and the other one for the averaged loss (on train/valid/test). You should be able to get less than 20% test error.