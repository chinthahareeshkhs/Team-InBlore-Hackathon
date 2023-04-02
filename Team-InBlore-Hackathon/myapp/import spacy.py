from fuzzywuzzy import fuzz

# define the available questions and corresponding functions
questions = {
    "rename a file in a folder with timestamp": "rename_with_timestamp",
    "rename a file in a folder with sequence": "rename_with_sequence",
    "rename a file in a folder with extension": "rename_with_extension",
    "alter filename of a file in a folder": "alter_filename",
}

# define the function to find the closest matching question


def find_closest_question(query, questions):
    # initialize the best question and score
    best_question = None
    best_score = 0

    # iterate through the available questions
    for question in questions:
        # compute the similarity score between the query and the question
        score = fuzz.token_set_ratio(query, question)

        print(score, " = ", best_score)

        # if the score is higher than the current best score, update the best question and score
        if score > best_score:
            best_question = question
            best_score = score

    # return the function corresponding to the closest matching question

    if (best_score > 75):
        return questions.get(best_question)
    else:
        return "not found"


# example usage
query = "rename a file with extention"
function = find_closest_question(query, questions)
if function:
    print(function)
else:
    print("No matching question found")
