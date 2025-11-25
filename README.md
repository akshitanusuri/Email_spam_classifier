# nlp_final_project_10-12
         EMAIL SPAM CLASSIFIER README



First we imported the necessary libraries, reads a CSV file named email_spam.csv into a DataFrame (df) and randomly displays 15 rows from the dataset to give a quick overview of its contents.


Basic Data cleaning : 
	Here, in basic data cleaning, we need to remove all null values, duplicate values. We changed category which has ham and spam as 0 and 1 respectively for better processing.

	We imported LabelEncoder to change ham/spam to 0/1
         After dropping 415 duplicate values , our emails are


         reduced from 5572 to 5157. 
	For better representation of number of spam and ham emails , we represented them in form of a pie charted. For this we imported matplotlib.pyplot.

EDA:
In data analysis to install nltk library and for tokenizing text we used import nltk and nltk.download('punkt').

	Using nltk tokenization function to break message into words ,characters and sentences 

	Now we had three new columns that are no_characters,no_words and no_sentences

	We compared the mean of these columns for both Ham and Spam 

	Using describe function we separated spam and ham emails

	In the comparison of the mean spam and ham.Mean of no.of characters in ham is 70 and whereas, in spam is 137.By analysing it we can say for which email has more no.of characters or words is a spam email.

	We used seaborn as sns to show the difference in characters in spam and ham through histogram plots.

Data Preprocessing:

	In Data Preprocessing firstly the transform_Message function  takes a message, converts it to lowercase then it splits into words using nltk.word_tokenize() and removes unnecessary parts. Using Stemming the words will be reduced to their base form and finally returning the cleaned-up version of the message.

	We imported the list of English stopwords from NLTK and gets back from it and  also imported Python's string module to access a list of punctuation characters.

	We performed stemming on word “hiding” using the PorterStemmer and  reduces to its base form after it looks at the 244th message in the data table.

	We applied the function transform_Message to every entry in the "Message" column of the DataFrame and processes each message and then  stores the results in a new column called "transformed_Message". Next we created and displayed a word cloud showing the most common words in spam and non-spam messages from the DataFrame.

	We used df.head() function to display the first five rows of the DataFrame to provides a quick overview of its content.And collected all words from spam messages into a list and counts the total number of words.     

	And then created a bar chart showing the 50 most common words in the spam messages by taking the word frequency on the y-axis and the word on the x-axis.

	We created a list of all words from non-spam messages  and calculated the total number of words. And then creates a bar chart showing the 50 most common words in non-spam messages  having word frequency on the y-axis and the words on the x-axis.

Feature Engineering:

For vectorising we used CountVectorizer and TfidfVectorizer to convert the transformed messages in the DataFrame into numerical features and  preparing the data for ML and then stores the result in an array X. Here X shape indicates the number of messages and features.Next extracts the category labels from the DataFrame and stores them as a NumPy array, y.

Data Modeling:
 
	Now in Data Modeling we split the dataset into training and testing sets and then uses  GaussianNB, MultinomialNB,  BernoulliNB classifiers to classify the messages. We trains each Naive Bayes model using the training data and tests their performance on unseen data and then evaluates their accuracy, confusion matrix, precision score using the test data.

	Next we sets up various machine learning models and stores them in a dictionary. We also defines a function train_classifier that trains a given model on the training data, tests it on the test data and returns its accuracy and precision scores. 

	Now we evaluates each model in clfs, calculates accuracy and precision for each and stores these scores in separate lists.

	We created a performance_df  DataFrame to store and classifys the performance  of each classifier from clfs.And now we reshapes the performance_df DataFrame from a wide format to a long format by  making it easier to  analyze the performance metrics by algorithm.

	We created a sns.catplot for bar plot to compare the accuracy and precision scores of different algorithms by helping to identify the best-performing model.

Model Improvement:

	We created  multiple DataFrames  i.e for comparing algorithm performance under different conditions and merged them to analyze and compare results effectively.

	And then created  Voting Classifier that combines predictions from three models i.e SVC, Multinomial Naive Bayes, Extra Trees Classifier using soft voting where probabilities are averaged. After that we trains the combined model on the training data.

	Now we evaluated a Voting Classifier and a Stacking Classifier, then calculated their accuracy and precision on test data and saves the TF-IDF vectorizer and Multinomial Naive Bayes model using pickle for future use.





