## deep-learning-challenge
## Alphabet Soup Charity Neural Network Analysis

# Overview
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. Uisng the features of the dataset I created a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

# Data Preprocessing:
Target Variable:
The target variable that was used for this neural network model was the "IS_SUCCESSFUL" column. Since the goal of the analysis was to build a model that is able to accurately predict applicants that had the best chance of being success in their ventures, "IS_SUCCESSFUL" was the perfect target variable for achieving this goal.

Feature Variables:
The features that were used in the original model were "APPLICATION_TYPE", "AFFILIATION", "CLASSIFICATION", "USE_CASE", "ORGANIZATION", "STATUS", "INCOME_AMT", "SPECIAL_CONSIDERATIONS", and "ASK_AMT". These features gave diverse information that the model could find complex relationships between to help build the weights between each node to give accurate predictions.

Irrelevant Data:
The final two columns of data that were included in the dataset that were not used as features for the original model was the "EIN" and "NAME" column. 

# Compiling, Training, and Evaluating the Model:
Neurons, Layers, and Activation Functions Used for Neural Network Model:

Neurons:
The original model utilized 80 neurons in the first hidden layer and 30 neurons in the second hidden layer with a single neuron in the output layer. This configuration achieved the highest accuracy with the lowest loss function value after exploring various architectures. This approach maximizes computational efficiency.

Layers:
The original model architecture employs two hidden layers and a single output layer. This configuration was chosen after considering various architectures. Two hidden layers provide a balance between model complexity and capacity. A single hidden layer might struggle to capture the intricacies of the data, while a significantly deeper architecture could lead to overfitting.

Activation Functions:
The original model employs ReLU (Rectified Linear Unit) activation functions in the two hidden layers and a sigmoid activation function in the output layer. ReLU offers advantages like computational efficiency and the ability to address the vanishing gradient problem that can hinder training in deeper networks. Since this model is a binary classification task aiming to predict successful (1) or unsuccessful (0) outcomes, a sigmoid activation function is appropriate due to sigmoid outputs values between 0 and 1, which can be interpreted as probabilities of an application being successful.

Target Model Performance:
The target model performance was 75% accuracy before the model was built. The original model was 73.06% which did not meet the standards that Alphabet Soup was expecting. This meant that the model had to be optimized to try and reach the goal of atleast 75% accuracy before the model can be put to use.

Steps Taken to Increase Model Performance / Optimization of Model:

Step 1: The first step that was taken to increase the performance of the model was to use more layers. My first attempt at compiling a neuron network consisted of 70, 50 and 30 neurons in the first, second and third layers. All layers had relu activation functions and the output layer had a sigmoid activation function. I started with these parameters as relu does better with nonlinear data, and two layers allows for a second layer to reweight the inputs from the first layer. Here are the preformance metrics of this model.

    hidden layers:
    nn_morelayers.add(tf.keras.layers.Dense(units=70, activation='relu'))
    nn_morelayers.add(tf.keras.layers.Dense(units=50, activation='relu'))
    nn_morelayers.add(tf.keras.layers.Dense(units=30, activation='relu'))

    Total params: 16,541 (64.61 KB)

    output:
    215/215 - 0s - 1ms/step - accuracy: 0.7278 - loss: 0.5827
    Loss: 0.5826636552810669, Accuracy: 0.7278425693511963


Step 2: The second step that was taken to increase the performance of the model was to replace the layers of relu with tanh. My second attempt at compiling a neuron network consisted of 70, 50 and 30 neurons in the first, second and third layers. All layers had tanh activation functions and the output layer had a sigmoid activation function. Here are the preformance metrics of this model.

    hidden layers:
    nn_tanh.add(tf.keras.layers.Dense(units=70, activation='tanh'))
    nn_tanh.add(tf.keras.layers.Dense(units=50, activation='tanh'))
    nn_tanh.add(tf.keras.layers.Dense(units=30, activation='relu'))

    Total params: 16,541 (64.61 KB)

    output:
    215/215 - 1s - 3ms/step - accuracy: 0.7289 - loss: 0.5678
    Loss: 0.5677856206893921, Accuracy: 0.728863000869751

Step 3: The third step that was taken was to mix the model with both relu and tanh activation functions with two hidden layers of tanh and one layer of relu to the model and experimenting with the neurons for each layer to see which combination gave the best performance. (First Hidden Layer => 75 neurons, Second Hidden Layer => 70 neurons, Third Hidden Layer => 65 neurons).

    hidden layers
    nn_morelayers_moreneurons.add(tf.keras.layers.Dense(units=75, activation='relu'))
    nn_morelayers_moreneurons.add(tf.keras.layers.Dense(units=70, activation='relu'))
    nn_morelayers_moreneurons.add(tf.keras.layers.Dense(units=65, activation='relu'))

    Total params: 21,836 (85.30 KB)

    output:
    215/215 - 0s - 1ms/step - accuracy: 0.7278 - loss: 0.6034
    Loss: 0.6033722162246704, Accuracy: 0.7278425693511963


# Summary
Overall Results:
The model did a good job of predicting Alphabet Soup applicants with the best chances of success in their ventures. Even after optimizing the model, I was unable to create a model that could preform a 75% accuracy rating. Overall, a suitable model was created for Alphabet Soup and although the model closely meets their requirments (73.06%), it can still be improved. Given the complexity of the data, a neural network model appears to be a well-suited choice for Alphabet Soup's use case. Applicant success after receiving funding likely hinges on a multitude of factors that may not be entirely captured in the current dataset. The features and information used to train the model could exhibit intricate relationships that a neural network is adept at recognizing compared to simpler classification models like Logistic Regression, Random Forests, or K-Nearest Neighbors. While there's always room for improvement in the current model's accuracy, for Alphabet Soup's specific use case, the inherent complexity of the data and the potential for non-linear relationships between features make neural networks a compelling option for identifying the most relevant factors associated with successful ventures.
