from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


# load the dataset containing user inputs and their corresponding questions
data = pd.read_csv(
    '/home/i2240/Downloads/djangotemplates-master/testing/myapp/sheet1.csv')

# create a TF-IDF vectorizer and fit it to the questions
vectorizer = TfidfVectorizer()
vectorizer.fit(data['question'])

# define a function to determine the closest matching question for a given user input


def find_closest_question(query, data, threshold=0.8):
    # tokenize the query and remove stop words
    query_tokens = [word.lower() for word in word_tokenize(
        query) if word.lower() not in stopwords.words('english')]

    # compute the Jaccard similarity between the query and each question
    similarities = []
    for question in data['question']:
        question_tokens = [word.lower() for word in word_tokenize(
            question) if word.lower() not in stopwords.words('english')]
        similarity = len(set(query_tokens).intersection(
            question_tokens)) / len(set(query_tokens).union(question_tokens))
        similarities.append(similarity)

    # find the index of the question with the highest similarity score
    closest_index = max(range(len(similarities)), key=similarities.__getitem__)

    # get the similarity score for the closest matching question
    closest_score = similarities[closest_index]

    # return the corresponding answer if the similarity score is above the threshold
    print(similarities, closest_score)
    if closest_score > threshold:
        return data.loc[closest_index, 'answer']
    else:
        return None


# example usage
query = "write a python"
closest_question = find_closest_question(query, data)
print(closest_question)
if closest_question:
    print(f"The closest matching question is: {closest_question}")
else:
    print("No matching question found")
