# from flask import Flask, jsonify, request 
# import torch 
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, pipeline

# app = Flask(__name__) 

# model_name = "bert-base-uncased"
# config = AutoConfig.from_pretrained(model_name)
# config.num_labels = 4
# tokenizer = AutoTokenizer.from_pretrained(model_name) 
# model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True) 
# labels = ["Risk", "Development Team", "Requirements", "User Participation"] 
# increments = 5 
# print(model.classifier.out_features)


# # Define the questions for each label
# questions = {
#     "Development Team": ["Is there a clear definition of roles and responsibilities?", "Is the team trained and qualified for the project?"],
#     "Requirements": ["Are the requirements documented?", "Are the requirements clear and unambiguous?", "Are the requirements complete?"],
#     "Risk": ["The project is the improvement of the old system?", "The requirements are highly reliable?"],
#     "User Participation": ["The users participate in all the phases?", "Is there limited user participation?"],
#     # Add questions for the other labels here
# }

# def generate_predictions(text, labels, increments): 
#     inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt") 
#     outputs = model(**inputs) 
#     logits = outputs.logits.detach().numpy()[0] 
#     # logits = logits.reshape(1, -1) # reshape to (1, 4)
#     probabilities = torch.softmax(torch.tensor(logits), dim=0).tolist() 
#     results = {} 
#     print(logits.shape)
#     print("1") 
#     print(probabilities)
#     for i, label in enumerate(labels): 
#         if i < len(probabilities): 
#             rounded_value = round(probabilities[i] * increments) / increments
#             # rounded_value = [round(p * increments) / increments for p in probabilities]
 
#             results[label] = rounded_value 
#         else: 
#             results[label] = 0 
#     print("2")
#     print(results)
#     return results 

# def generate_answers(text, label):
#     # Load model for label-specific questions
#     # model_name = "bert-base-uncased"
#     # tokenizer = AutoTokenizer.from_pretrained(model_name)
#     # model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
#     questions_list = questions[label]
#     answers = []
#     print(questions_list)
#     for question in questions_list:
#         inputs = tokenizer(question, text, padding=True, truncation=True, return_tensors="pt")
#         outputs = model(**inputs)
#         prediction = torch.argmax(outputs.logits).item()
#         if prediction == 0:
#             answers.append("no")
#         else:
#             answers.append("yes")
#     print(answers)
#     return answers

# @app.route('/') 
# def index(): 
#     return open('index.html').read() 

# @app.route('/submit', methods=['POST']) 
# def submit(): 
#     # Get the text input from the form 
#     text = request.form.get('text') 
#     if not text: 
#         return jsonify({'error': 'Empty input'}), 400 
#    # Print the text in Python 
#     sentences = text.split('.')  # Split on . for sentences 
#     results = [] 
#     for sentence in sentences:
#         result = generate_predictions(sentence, labels, increments)
#         for label, value in result.items():
#             if value > 0:
#                 answers = generate_answers(sentence, label)
#                 results.append({label: answers})

#     return jsonify(results)

# if __name__ == '__main__': 
#     app.run()



from flask import Flask, jsonify, request 
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, pipeline

app = Flask(__name__) 

print(app)

model_name = "bert-base-uncased"
config = AutoConfig.from_pretrained(model_name)
config.num_labels = 4
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True) 
labels = ["Risk", "Development Team", "Requirements", "User Participation"] 
increments = 5 
print(model.classifier.out_features)


# Define the questions for each label Knowledge in Domain is Little
questions = {
    "Development Team": ["The team has experience on similar projects?", "Is the team knowledgeable about the domain of the project?", "The team has experience with the tools used for the project?", "There is training available for the team?"],
    "Requirements": ["Easy to understandable and defined requirements?", "Requirements are changed quite often?", "Requirements are defined early in the cycle?", "Is the system complex due to the requirements?"],
    "Risk": ["The project is the improvement of an old system?", "The requirements are highly reliable?", "Is there stable funding for the project?", "Is the schedule of the project tight?", "Can reusable components be used?", "Are there scare resources for the project?"],
    "User Participation": ["The users participate in all the phases?", "Is there limited user participation?", "Does the user have any experience on similar projects?", "Are the users experts of the problem domain?"],
    # Add questions for the other labels here
}

def generate_predictions(text, labels, increments): 
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt") 
    outputs = model(**inputs) 
    logits = outputs.logits.detach().numpy()[0] 
    probabilities = torch.softmax(torch.tensor(logits), dim=0).tolist() 
    results = {} 
    for i, label in enumerate(labels): 
        if i < len(probabilities): 
            rounded_value = round(probabilities[i] * increments) / increments
            results[label] = rounded_value 
        else: 
            results[label] = 0 
    print(results)
    return results 

def generate_answers(text, label):
    questions_list = questions[label]
    print(questions_list)
    answers = []
    for question in questions_list:
        inputs = tokenizer(question, text, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits).item()
        print(prediction)
        if prediction == 0:
            answers.append("Yes")
        else:
            answers.append("No")
    print(answers)
    return answers

@app.route('/') 
def index(): 
    return open('index.html', 'r').read()

@app.route('/submit', methods=['POST'])
def submit():
    text = request.form['text']
    predictions = generate_predictions(text, labels, increments)
    results = []
    for label in labels:
        answers = generate_answers(text, label)
        results.append({'label': label, 'answers': answers})
    print(predictions)
    print(results)
    return jsonify({'probabilities': [predictions], 'results': results})

@app.route('/second_submit', methods=['POST'])
def second_submit():
    data = request.get_json()
    del data['probabilities']
    results = data["results"]
    print(results)
    # Process the data from the second form submission
    # Define the SDLCs and their corresponding scores
    sdlcs = {
        "Waterfall": 0,
        "Prototype": 0,
        "Iterative": 0,
        "Spiral": 0,
        "RAD": 0,
        "XP": 0
    }

    sdlc_responses = { # risk   dev team    requirments     user
    'Waterfall': ['no', 'no', 'yes', 'no', 'no', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'no', 'yes', 'no', 'no'],
    'Prototype': ['no', 'no', 'yes','yes','yes','yes', 'yes', 'no', 'no', 'no', 'no', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'yes'],
    'Iterative': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes'],
    'Spiral': ['no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'no'],
    'Rad': ['yes', 'no', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'no', 'yes'],
    'Xp': ['yes', 'yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
    
# initialize list to hold user responses
    user_responses = []

    # get user responses from data
    for label in data['results']:
        
        # loop through each answer for the current label
        for answer in label['answers']:
            
            # append the answer (converted to lowercase) to the user_responses list
            user_responses.append(answer.lower())

    print("User Responses:")
    print(user_responses)
    # calculate the score for each SDLC
    scores = {}
    for sdlc, responses in sdlc_responses.items():
        count = 0
        for i, response in enumerate(responses):
            if response.lower() == user_responses[i]:
                count += 1
        scores[sdlc] = count

    # print scores
    print("Scores:")
    print(scores)

    percentages = {}
    for sdlc, count in scores.items():
        percentage = count / 18 * 100
        percentages[sdlc] = round(percentage, 1)

    # sort the SDLCs by their percentage match
    sorted_sdlcs = sorted(percentages.items(), key=lambda x: x[1], reverse=True)

    # Find the SDLC with the highest score
    best_sdlc = max(scores)
    print("Best_sdlc:")
    print(best_sdlc)
    print("scores:")
    print(scores)
    print("sorted_sdlcs")
    print(sorted_sdlcs)
    # Return the data and the best SDLC
    return jsonify({'sorted_SDLCs': sorted_sdlcs})



if __name__ == '__main__':
    app.run()
