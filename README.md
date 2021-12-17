# Covid-19-Chatbot
The Covid-19 pandemic has hit hard globally, compelling us to think about the future of human species, its civilisation and its fight against diseases. There has always been shortage and misinformation regarding the spread of covid-19. People are not well informed about the necessary precautions, safety measures and protocols. Instead they had to randomly search in different web pages for informations and the essentials, making it very time consuming. But in this modern era we have abundance of technologies and here we have utilise them to make information easily available to anyone in need. Anyone can have different issues and question about coronavirus, and so, in this project we tried to implement a Covid-19 Chatbot for easier access to information. The chatbot would simulate human-like conversation and thus ultimately be a good and useful tool for every user. The chatbot will give responses to each and every different queries of every users efficiently. This is what we wanted to achieve through this project.
Covid 19 chatbot
What is a chatbot ?
A chatbot is an intelligent piece of software that is capable of communicating and performing actions similar to a human.
It actually simulates human-like conversations with users via text messages on chat. Chatbots are of two types based on how they are built:-
1. Retrieval based
2. Generative based models.
Here for this project, we made a retrieval based chatbot. A retrieval-based chatbot uses predefined input patterns and responses.
That is, a retrieval based chatbot already has the required output actions/response in advance.
It then uses some type of heuristic approach to select the appropriate response. But the generative based models are not based on some predefined responses.
They require a huge amount of data.

About Dataset:- 
We prepared the dataset for this project. The dataset actually consists of random responses from the list of responses. One response was like-
If the input sentence from the user contains the word “vaccine”, or something related to “vaccine”,
then the response from the chatbot would be the link to the cowin website of the government from where the details of vaccine can be fetched out.
Further, if user inputs somewhat like healthy foods, the chatbot would produce a response of healthy foods rich with vitamin C.
So, the overall functioning of chatbot was good, as it can give users a lot of information regarding covid 19 and its precautions.

The dataset consists of 3 parts:- 
1)tags 2)patterns 3)responses
-tags means under what category the output of chatbot should be classified.
-patterns means the input sentences of users. So the main task is to preprocess the input sentences of users and use classification algorithms like naive bayes,
or logistic regression, or either use deep learning techniques to classify the tags of the patterns.
-response means what the chatbot should output. It consists of random sentences related to the tags.
For example:-
"tag": "precautions",
"patterns": ["What are the necessary precautions for corona avoidance?", "Precautions for covid19", "What should we do to avoid corona?", "Is corona avoidable ?"],
"responses": ["Maintain social distance- 2 feet apart, Avoid touching eyes and mouth, Sanitise your hands properly"],

We made the chatbot in 5 steps:-
1. Import and load the data file
2. Preprocess data
3. Create training and testing data
4. Build the model
5. Predict the response

-----1. Import and load the data file:
We have already said about the dataset. The dataset is stored in json format. So we used the json package of python to load the dataset.
-----2. Preprocess data:
It is one of the main processes. For any textual file, the main steps of preprocessing are tokenizing the sentences,lemmatizing the words,
and removing stop words. These are some of the basic nlp tasks to be done in the text file. In simple words, tokenizing means breaking the sentences into words.
Lemmatization means converting a word to its base or root form.
For example:- “studies” to “study”, “eating” to “eat”.
Removing stop words like “is” , “am” , etc are also important as these stop words do not carry a meaning in the sentences.
This preprocess step is very very crucial as our computers directly cannot work with text files.
It needs data in the form of numbers so after tokenizing and lemmatizing we will have to convert it into numbers for our ML model to work upon.
It will be done in the 3rd step, for tokenizing and lemmatizing, we have used the nltk library(nltk = natural language toolkit). 
It is one of the most important libraries in the field of nlp , and is widely used for research.
-----3.Create training and testing data:
Now, we will create the training data in which we will provide the input and the output. Our input will be the pattern(as said in the dataset)
and output will be the class(tags) our input pattern belongs to. So basically it is a classification task for us. 
Classification means in which category an input belongs to. Famous classification algorithms are naive bayes, logistic regression etc. 
But here we used deep learning techniques for classification. 
We have used ANN(artificial neural networks) to classify which category the user’s message belongs to and
then we will give a random response from the list of responses. In place of ANN, we could have used RNN(recurrent neural network) too.
In all ML tasks, there are three types of dataset- training, testing, and validation/development dataset. 
But here, we have not made a validation dataset as the size of the dataset is small.
-----training dataset - use to train the model
-----validation dataset - used for hyperparameter tuning like number of neighbours in Kth nearest neighbours(knn) algorithm, or learning rate in neural networks.
Generally, it is used to check whether the model has overfitted or not on the training dataset. If overfitting occurs, that means the model has lost the accuracy, it has high
accuracy on the training dataset but on the test dataset it fails, i.e., it has low accuracy on the test dataset. 
It means it has lost or failed to understand the underlying logic of the data.
-----testing dataset - used to check the accuracy of the model on the unknown dataset which the model has not seen before during the training process.

--- Some notes ---
Overfitting and underfitting:-
Overfitting is already written earlier. Underfitting means the model is not trained well on the dataset and will give error outputs,
and hence is not a good model on the dataset.
Both overfitting and underfitting are bad for the model.
-----4. Build the model: We have our training data ready, now we will build a deep neural network that has 3 layers. We use the Keras sequential API for this.
After training the model for 200 epochs, we achieved 100% accuracy on our model. We achieved such an accuracy as the size of the dataset is small, 
otherwise majority models have an accuracy of 92-93%.
Epochs means the number of times the model will be trained on the training dataset.
Our created model has 3 layers. 
First layer 128 neurons, second layer 64 neurons and 3rd output layer contain the number of neurons equal to the number of tags in the input dataset.
We have used the softmax activation function in the 3rd output layer to predict the tags of the incoming user sentences. 
Softmax activation function is used for classification tasks in deep neural networks. It actually returns the probability of each tag for the incoming user sentences,
and then the tag which has the highest probability will be chosen as the tag of the user input. 
And from the response, we will select any random sentences related to the text.
We then saved the model after training it for 200 epochs.
-----5. Predict the response: We will load the saved model and then use a graphical user interface(tkinter) that will predict the response from the bot.
The model will only tell us the class it belongs to, so we will implement some functions which will identify the class and then retrieve us a random
response from the list of responses.
