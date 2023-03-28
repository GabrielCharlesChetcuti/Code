from flask import Flask, jsonify, request 
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
app = Flask(__name__) 
model_name = "bert-base-uncased"
config = AutoConfig.from_pretrained(model_name)
config.num_labels = 4
# labels2 = ["Development Team", "Requirements"]
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True) 
labels = ["Risk", "Development Team", "Requirements", "Stakeholder Involvement"] 
increments = 5 
print(model.classifier.out_features)

def generate_predictions(text, labels, increments): 
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt") 
    outputs = model(**inputs) 
    logits = outputs.logits.detach().numpy()[0] 
    # logits = logits.reshape(1, -1) # reshape to (1, 4)
    probabilities = torch.softmax(torch.tensor(logits), dim=0).tolist() 
    results = {} 
    print(logits.shape)
    print("1") 
    print(probabilities)
    for i, label in enumerate(labels): 
        if i < len(probabilities): 
            rounded_value = round(probabilities[i] * increments) / increments
            # rounded_value = [round(p * increments) / increments for p in probabilities]
 
            results[label] = rounded_value 
        else: 
            results[label] = 0 
    print("2")
    print(results)
    return results 

@app.route('/') 
def index(): 
    return open('index.html').read() 

@app.route('/submit', methods=['POST']) 
def submit(): 
    # Get the text input from the form 
    text = request.form.get('text') 
    if not text: 
        return jsonify({'error': 'Empty input'}), 400 
   # Print the text in Python 
    sentences = text.split('.')  # Split on . for sentences 
    results = [] 
    for sentence in sentences: 
        result = generate_predictions(sentence, labels, increments) 
        print(sentence)
        results.append(result) 
    print("3")
    print(results)
    return jsonify(results)

if __name__ == '__main__': 
    app.run()
