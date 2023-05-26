from flask import Flask, jsonify, request 
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, pipeline

app = Flask(__name__) 
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

print(app)

model_name = "bert-base-uncased"
config = AutoConfig.from_pretrained(model_name)
config.num_labels = 4
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True) 
labels = ["Risk", "Development Team", "Requirements", "User Participation"] 
increments = 5 
print(model.classifier.out_features)


# Define the questions for each label 
questions = {
    "Risk": ["The project is the improvement of an old system?", "The requirements are highly reliable?", "Is there stable funding for the project?", "Is the schedule of the project tight?", "Can reusable components be used?", "Are there scare resources for the project?"],
    "Development Team": ["The team has experience on similar projects?", "Is the team knowledgeable about the domain of the project?", "The team has experience with the tools used for the project?", "There is training available for the team?"],
    "Requirements": ["Easy to understandable and defined requirements?", "Requirements are changed quite often?", "Requirements are defined early in the cycle?", "Is the system complex due to the requirements?"],
    "User Participation": ["The users participate in all the phases?", "Is there limited user participation?", "Does the user have any experience on similar projects?", "Are the users experts of the problem domain?"],
}

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
    results = []
    for label in labels:
        answers = generate_answers(text, label)
        results.append({'label': label, 'answers': answers})
    print(results)
    return jsonify({'results': results})


@app.route('/second_submit', methods=['POST'])
def second_submit():
    data = request.get_json()
    try:
        print("Received data:", data)
        sdlc_responses = { # risk   dev team    requirments     user
        'Waterfall': ['no', 'no', 'yes', 'no', 'no', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'no', 'yes', 'no', 'no'],
        'Prototype': ['no', 'no', 'yes','yes','yes','yes', 'yes', 'no', 'no', 'no', 'no', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'yes'],
        'Iterative': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes'],
        'Spiral': ['no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'no'],
        'RAD': ['yes', 'no', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'no', 'yes'],
        'XP': ['yes', 'yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'no', 'no', 'yes']
    }
        
        explanation_map = {
            "The project is the improvement of an old system?": {
                "Waterfall": {
                    "no": "The Waterfall model works well for new systems as it follows a structured and linear approach, making it suitable for projects with well-defined requirements.",
                    "yes": "The Waterfall model might not be the best fit for projects improving an old system, as it follows a linear approach that might not handle uncertainties and changes well."
                },
                "Prototype": {
                    "no": "The Prototype model is effective for new systems as it helps manage uncertainties and refine requirements through iterative prototyping.",
                    "yes": "While the Prototype model can be helpful for improving an old system, it is more suited for projects with less-defined requirements and higher uncertainty."
                },
                "Iterative": {
                    "yes": "The Iterative model works well for projects that improve an old system, as it allows for continuous improvements and adjustments through iterations.",
                    "no": "The Iterative model might not be necessary for new systems with well-defined requirements, as other models might handle the development process better."
                },
                "Spiral": {
                    "no": "The Spiral model is suitable for new systems, as it combines iterative development with risk analysis to manage uncertainties and refine the development process.",
                    "yes": "The Spiral model might not be the most efficient choice for projects improving an old system, as it focuses on risk analysis and iterative development that might not be required."
                },
                "RAD": {
                    "yes": "The RAD model works well for projects that improve an old system, as it focuses on quickly delivering working prototypes and handling changes effectively.",
                    "no": "The RAD model might not be the best fit for new systems with well-defined requirements, as it prioritizes rapid application development that might not be required."
                },
                "XP": {
                    "yes": "The XP model is effective for projects that improve an old system, as it emphasizes continuous integration, quick feedback, and the ability to handle changes.",
                    "no": "The XP model might not be necessary for new systems with well-defined requirements, as its focus on quick feedback and continuous integration might not be required."
                }
            },
            "The requirements are highly reliable?": {
                "Waterfall": {
                    "no": "The Waterfall model might work well when requirements are less reliable, as it follows a structured and linear approach that can benefit from having well-defined requirements.",
                    "yes": "The Waterfall model might not be the best fit when requirements are highly reliable, as its linear approach might not handle changes and uncertainties well."
                },
                "Prototype": {
                    "no": "The Prototype model is effective when requirements are less reliable, as it helps manage uncertainties and refine requirements through iterative prototyping.",
                    "yes": "The Prototype model might be less suited when requirements are highly reliable, as the need for iterative prototyping to refine requirements might be reduced."
                },
                "Iterative": {
                    "yes": "The Iterative model works well when requirements are highly reliable, as it allows for continuous improvements and adjustments through iterations that can handle changes and uncertainties.",
                    "no": "The Iterative model might not be as necessary when requirements are less reliable, as other models might handle the development process better."
                },
                "Spiral": {
                    "yes": "The Spiral model is suitable when requirements are highly reliable, as it combines iterative development with risk analysis to manage uncertainties and refine the development process.",
                    "no": "The Spiral model might not be the most efficient choice when requirements are less reliable, as its focus on risk analysis and iterative development might not be as valuable."
                },
                "RAD": {
                    "no": "The RAD model might work well when requirements are less reliable, as it focuses on quickly delivering working prototypes and handling changes effectively.",
                    "yes": "The RAD model might not be the best fit when requirements are highly reliable, as its focus on rapid application development might not be as valuable."
                },
                "XP": {
                    "yes": "The XP model is effective when requirements are highly reliable, as it emphasizes continuous integration, quick feedback, and the ability to handle changes.",
                    "no": "The XP model might not be as necessary when requirements are less reliable, as its focus on quick feedback and continuous integration might not be as valuable."
                }
            },
            "Is there stable funding for the project?": {
                "Waterfall": {
                    "yes": "The Waterfall model works well with stable funding as it follows a structured and linear approach that requires a predictable budget allocation.",
                    "no": "The Waterfall model might not be the best fit when funding is not stable, as its linear approach and fixed budget allocations might not handle uncertainties well."
                },
                "Prototype": {
                    "yes": "The Prototype model is effective when there is stable funding, as it allows for iterative prototyping and refinement of requirements with a predictable budget.",
                    "no": "The Prototype model might not be the best choice when funding is not stable, as the iterative prototyping process might face budget constraints."
                },
                "Iterative": {
                    "no": "The Iterative model can work well when funding is not stable, as it allows for continuous improvements and adjustments through iterations that can adapt to budget changes.",
                    "yes": "The Iterative model might be less necessary when funding is stable, as other models might handle the development process better."
                },
                "Spiral": {
                    "no": "The Spiral model is suitable when funding is not stable, as it combines iterative development with risk analysis to manage uncertainties, including budget-related risks.",
                    "yes": "The Spiral model might not be the most efficient choice when funding is stable, as its focus on risk analysis and iterative development might not be as valuable."
                },
                "RAD": {
                    "yes": "The RAD model works well when there is stable funding, as it focuses on quickly delivering working prototypes and requires a predictable budget to support rapid application development.",
                    "no": "The RAD model might not be the best fit when funding is not stable, as its focus on rapid application development might face budget constraints."
                },
                "XP": {
                    "no": "The XP model is effective when funding is not stable, as it emphasizes continuous integration, quick feedback, and the ability to handle changes, including budget-related changes.",
                    "yes": "The XP model might not be as necessary when funding is stable, as its focus on quick feedback and continuous integration might not be as valuable."
                }
            },
            "Is the schedule of the project tight?": {
                "Waterfall": {
                    "no": "The Waterfall model is a better fit when the schedule is not tight, as it follows a linear and structured approach that may not be flexible enough to accommodate tight deadlines.",
                    "yes": "The Waterfall model may not be the best choice when the schedule is tight, as its inflexible structure and sequential phases might not be able to handle time constraints effectively."
                },
                "Prototype": {
                    "yes": "The Prototype model works well when the schedule is tight, as it focuses on building and refining prototypes quickly to reach a final product faster.",
                    "no": "The Prototype model may be less necessary when the schedule is not tight, as the time spent on iterative prototyping might not be as valuable."
                },
                "Iterative": {
                    "yes": "The Iterative model is suitable when the schedule is tight, as it allows for continuous improvements and adjustments through iterations that can adapt to time constraints.",
                    "no": "The Iterative model might be less necessary when the schedule is not tight, as other models might be more suitable for managing the development process."
                },
                "Spiral": {
                    "yes": "The Spiral model works well when the schedule is tight, as it combines iterative development with risk analysis to manage uncertainties and reach a final product faster.",
                    "no": "The Spiral model may be less efficient when the schedule is not tight, as its focus on risk analysis and iterative development might not be as valuable."
                },
                "RAD": {
                    "yes": "The RAD model is effective when the schedule is tight, as it focuses on quickly delivering working prototypes and rapidly reaching a final product.",
                    "no": "The RAD model might not be the best fit when the schedule is not tight, as its emphasis on rapid application development might not be as valuable."
                },
                "XP": {
                    "no": "The XP model is suitable when the schedule is not tight, as it emphasizes continuous integration and quick feedback, which might be better suited for projects without tight time constraints.",
                    "yes": "The XP model may not be as effective when the schedule is tight, as its focus on continuous integration and quick feedback might not be as valuable under time pressure."
                }
            },
            "Can reusable components be used?": {
                "Waterfall": {
                    "no": "The Waterfall model is not focused on reusing components, as it follows a linear and structured approach that requires specific requirements and designs for each phase.",
                    "yes": "The Waterfall model might not be the best choice when reusable components can be used, as its linear and structured approach might not fully leverage the benefits of reusability."
                },
                "Prototype": {
                    "yes": "The Prototype model is suitable for projects with reusable components, as it focuses on building and refining prototypes that can benefit from existing components.",
                    "no": "The Prototype model might be less efficient when reusable components cannot be used, as its focus on iterative prototyping may not fully leverage the benefits of reusability."
                },
                "Iterative": {
                    "no": "The Iterative model is not designed to focus on reusing components, as it emphasizes continuous improvements and adjustments through iterations.",
                    "yes": "The Iterative model might not be the best choice when reusable components can be used, as its emphasis on continuous improvements might not fully leverage the benefits of reusability."
                },
                "Spiral": {
                    "yes": "The Spiral model works well when reusable components can be used, as it combines iterative development with risk analysis and can benefit from existing components to manage uncertainties.",
                    "no": "The Spiral model may be less efficient when reusable components cannot be used, as its focus on risk analysis and iterative development might not fully leverage the benefits of reusability."
                },
                "RAD": {
                    "yes": "The RAD model is effective when reusable components can be used, as it focuses on quickly delivering working prototypes that can benefit from existing components.",
                    "no": "The RAD model might not be the best fit when reusable components cannot be used, as its emphasis on rapid application development might not fully leverage the benefits of reusability."
                },
                "XP": {
                    "no": "The XP model is not focused on reusing components, as it emphasizes continuous integration and quick feedback that may not fully leverage the benefits of reusability.",
                    "yes": "The XP model might not be the best choice when reusable components can be used, as its focus on continuous integration and quick feedback might not fully leverage the benefits of reusability."
                }
            },
            "Are there scarce resources for the project?": {
                "Waterfall": {
                    "no": "The Waterfall model might not be the best choice for projects with scarce resources, as it requires a structured and linear approach with dedicated resources for each phase.",
                    "yes": "The Waterfall model is more suitable for projects with ample resources, as it follows a structured and linear approach that requires specific resources for each phase."
                },
                "Prototype": {
                    "yes": "The Prototype model can be a good fit for projects with scarce resources, as it focuses on building and refining prototypes, which allows for more flexibility in resource allocation.",
                    "no": "The Prototype model might be less effective when resources are not scarce, as other models may better utilize the available resources."
                },
                "Iterative": {
                    "no": "The Iterative model might not be the best choice for projects with scarce resources, as it requires continuous improvements and adjustments through iterations, which may consume significant resources.",
                    "yes": "The Iterative model is more suitable for projects with ample resources, as it emphasizes continuous improvements and adjustments that require specific resources for each iteration."
                },
                "Spiral": {
                    "yes": "The Spiral model is a good fit for projects with scarce resources, as it combines iterative development with risk analysis, allowing for more efficient resource allocation and management.",
                    "no": "The Spiral model might be less effective when resources are not scarce, as other models may better utilize the available resources."
                },
                "RAD": {
                    "no": "The RAD model might not be the best choice for projects with scarce resources, as it emphasizes rapid application development, which may consume significant resources.",
                    "yes": "The RAD model is more suitable for projects with ample resources, as it focuses on quickly delivering working prototypes that require dedicated resources."
                },
                "XP": {
                    "no": "The XP model might not be the best choice for projects with scarce resources, as it emphasizes continuous integration and quick feedback, which may consume significant resources.",
                    "yes": "The XP model is more suitable for projects with ample resources, as it focuses on continuous integration and quick feedback that require specific resources for each iteration."
                }
            },
            "The team has experience on similar projects?": {
                "Waterfall": {
                    "no": "Lack of experience on similar projects might make the Waterfall model more challenging to implement, as it requires a well-defined and linear approach.",
                    "yes": "Having experience on similar projects is beneficial when using the Waterfall model, as it can help the team follow the well-defined and linear approach more effectively."
                },
                "Prototype": {
                    "yes": "Experience on similar projects is advantageous for the Prototype model, as the team can better build and refine prototypes based on their previous knowledge.",
                    "no": "Lack of experience on similar projects may make the Prototype model more difficult to implement, but it can still be valuable for learning and adapting to the project's needs."
                },
                "Iterative": {
                    "no": "Lack of experience on similar projects might make the Iterative model more challenging to implement, as it requires continuous improvements and adjustments through iterations.",
                    "yes": "Having experience on similar projects is beneficial when using the Iterative model, as it can help the team effectively adapt and improve through each iteration."
                },
                "Spiral": {
                    "yes": "Experience on similar projects is advantageous for the Spiral model, as the team can better manage risks and make informed decisions throughout the development process.",
                    "no": "Lack of experience on similar projects may make the Spiral model more difficult to implement, but its risk-driven approach can help the team learn and adapt to the project's needs."
                },
                "RAD": {
                    "no": "Lack of experience on similar projects might make the RAD model more challenging to implement, as it requires rapid application development and quick delivery of working prototypes.",
                    "yes": "Having experience on similar projects is beneficial when using the RAD model, as it can help the team quickly deliver working prototypes and adapt to the project's needs."
                },
                "XP": {
                    "no": "Lack of experience on similar projects might make the XP model more challenging to implement, as it emphasizes continuous integration and quick feedback.",
                    "yes": "Having experience on similar projects is beneficial when using the XP model, as it can help the team effectively integrate and adapt based on continuous feedback."
                }
            },
            "Is the team knowledgeable about the domain of the project?": {
                "Waterfall": {
                    "yes": "Domain knowledge is crucial for the Waterfall model, as it helps the team effectively plan and execute each stage of the linear process.",
                    "no": "Lack of domain knowledge might make the Waterfall model more challenging to implement, as it relies on a well-defined and linear approach that requires a deep understanding of the project's domain."
                },
                "Prototype": {
                    "no": "Lack of domain knowledge might not be a significant issue when using the Prototype model, as it allows the team to build and refine prototypes while learning more about the project's domain.",
                    "yes": "Having domain knowledge is beneficial when using the Prototype model, but it is not a strict requirement, as the team can still build and refine prototypes to learn more about the project's domain."
                },
                "Iterative": {
                    "yes": "Domain knowledge is advantageous for the Iterative model, as it helps the team effectively adapt and improve the project through each iteration.",
                    "no": "Lack of domain knowledge might make the Iterative model more challenging to implement, but its iterative approach allows the team to learn and adapt to the project's domain over time."
                },
                "Spiral": {
                    "yes": "Domain knowledge is beneficial for the Spiral model, as it helps the team manage risks and make informed decisions throughout the development process.",
                    "no": "Lack of domain knowledge might make the Spiral model more challenging to implement, but its risk-driven approach can help the team learn and adapt to the project's domain over time."
                },
                "RAD": {
                    "no": "Lack of domain knowledge might not be a significant issue when using the RAD model, as it emphasizes rapid application development and quick delivery of working prototypes, allowing the team to learn about the project's domain.",
                    "yes": "Having domain knowledge is beneficial when using the RAD model, as it can help the team quickly deliver working prototypes and adapt to the project's needs."
                },
                "XP": {
                    "no": "Lack of domain knowledge might not be a significant issue when using the XP model, as it emphasizes continuous integration and quick feedback, allowing the team to learn about the project's domain.",
                    "yes": "Having domain knowledge is beneficial when using the XP model, as it can help the team effectively integrate and adapt based on continuous feedback."
                }
            },
            "The team has experience with the tools used for the project?": {
                "Waterfall": {
                    "yes": "Experience with the tools used for the project is essential for the Waterfall model, as it relies on a linear and sequential process where tool proficiency helps ensure smooth progress.",
                    "no": "Lack of experience with the tools used for the project might make the Waterfall model more challenging to implement, as it relies on a linear and sequential process that requires proficiency with the tools."
                },
                "Prototype": {
                    "no": "Lack of experience with the tools used for the project might not be a significant issue when using the Prototype model, as it focuses on building and refining prototypes while the team learns the tools.",
                    "yes": "Having experience with the tools used for the project is beneficial when using the Prototype model, as it can help the team quickly build and refine prototypes."
                },
                "Iterative": {
                    "no": "Lack of experience with the tools used for the project might not be a significant issue when using the Iterative model, as the team can learn the tools throughout the iterative process.",
                    "yes": "Having experience with the tools used for the project is beneficial when using the Iterative model, as it can help the team effectively improve the project through each iteration."
                },
                "Spiral": {
                    "yes": "Experience with the tools used for the project is important for the Spiral model, as it helps the team manage risks and make informed decisions throughout the development process.",
                    "no": "Lack of experience with the tools used for the project might make the Spiral model more challenging to implement, but its risk-driven approach can help the team learn and adapt to the tools over time."
                },
                "RAD": {
                    "no": "Lack of experience with the tools used for the project might not be a significant issue when using the RAD model, as it emphasizes rapid application development and quick delivery, allowing the team to learn the tools.",
                    "yes": "Having experience with the tools used for the project is beneficial when using the RAD model, as it can help the team quickly deliver working prototypes and adapt to the project's needs."
                },
                "XP": {
                    "no": "Lack of experience with the tools used for the project might not be a significant issue when using the XP model, as it emphasizes continuous integration and quick feedback, allowing the team to learn the tools.",
                    "yes": "Having experience with the tools used for the project is beneficial when using the XP model, as it can help the team effectively integrate and adapt based on continuous feedback."
                }
            },
            "There is training available for the team?": {
                "Waterfall": {
                    "no": "The absence of training for the team might make the Waterfall model more challenging to implement, as it relies on a linear and sequential process where proficiency in tools and knowledge is essential.",
                    "yes": "Training for the team might not be as crucial for the Waterfall model, as it relies on a linear and sequential process where the team's existing knowledge and proficiency in tools are more important."
                },
                "Prototype": {
                    "no": "The absence of training for the team might not be a significant issue when using the Prototype model, as it focuses on building and refining prototypes while the team learns through hands-on experience.",
                    "yes": "Training for the team might not be as crucial for the Prototype model, as it focuses on learning through building and refining prototypes."
                },
                "Iterative": {
                    "yes": "Training for the team can be beneficial when using the Iterative model, as it allows the team to quickly adapt and improve the project through each iteration.",
                    "no": "The absence of training for the team might not be a significant issue when using the Iterative model, as the team can learn and adapt throughout the iterative process."
                },
                "Spiral": {
                    "no": "The absence of training for the team might make the Spiral model more challenging to implement, but its risk-driven approach can help the team learn and adapt over time.",
                    "yes": "Training for the team might not be as crucial for the Spiral model, as it emphasizes risk management and adaptation, allowing the team to learn on the job."
                },
                "RAD": {
                    "yes": "Training for the team can be beneficial when using the RAD model, as it helps the team quickly deliver working prototypes and adapt to the project's needs.",
                    "no": "The absence of training for the team might not be a significant issue when using the RAD model, as it emphasizes rapid application development and quick delivery, allowing the team to learn through hands-on experience."
                },
                "XP": {
                    "yes": "Training for the team can be beneficial when using the XP model, as it helps the team effectively integrate and adapt based on continuous feedback.",
                    "no": "The absence of training for the team might not be a significant issue when using the XP model, as it emphasizes continuous integration and quick feedback, allowing the team to learn on the job."
                }
            },
            "Easy to understandable and defined requirements?": {
                "Waterfall": {
                    "yes": "Clear and well-defined requirements are essential for the Waterfall model, as it relies on a linear and sequential process where changes are difficult to make once the project starts.",
                    "no": "If requirements are not easily understandable or well-defined, the Waterfall model might not be the best choice, as it relies on a linear and sequential process where changes are difficult to make once the project starts."
                },
                "Prototype": {
                    "no": "When requirements are not easily understandable or well-defined, the Prototype model can be useful, as it involves building and refining prototypes to clarify and finalize requirements.",
                    "yes": "If requirements are easily understandable and well-defined, the Prototype model might not be as necessary, as the focus on building and refining prototypes might not provide significant benefits."
                },
                "Iterative": {
                    "no": "When requirements are not easily understandable or well-defined, the Iterative model can be useful, as it allows for the project to progress through iterations, adapting and clarifying requirements along the way.",
                    "yes": "If requirements are easily understandable and well-defined, the Iterative model might not be the best choice, as other models might better fit the project."
                },
                "Spiral": {
                    "no": "When requirements are not easily understandable or well-defined, the Spiral model can be useful, as its risk-driven approach helps to manage uncertainty and adapt to changing requirements.",
                    "yes": "If requirements are easily understandable and well-defined, the Spiral model might not be the best choice, as other models might better fit the project."
                },
                "RAD": {
                    "yes": "Clear and well-defined requirements are beneficial for the RAD model, as it helps the team quickly deliver working prototypes and adapt to the project's needs.",
                    "no": "If requirements are not easily understandable or well-defined, the RAD model might not be the best choice, as it relies on rapid application development and quick delivery, which can be hindered by unclear requirements."
                },
                "XP": {
                    "no": "When requirements are not easily understandable or well-defined, the XP model can be useful, as it emphasizes continuous integration and quick feedback, allowing for clarification and adaptation of requirements.",
                    "yes": "If requirements are easily understandable and well-defined, the XP model might not be the best choice, as other models might better fit the project."
                }
            },
            "Requirements are changed quite often?": {
                "Waterfall": {
                    "no": "This is good for the Waterfall model as it prefers to have all the requirements defined at the beginning.",
                    "yes": "This is not good for the Waterfall model since it prefers to have the requirements defined at the beginning."
                },
                "Prototype": {
                    "yes": "Frequent requirement changes can be managed with the Prototype model as it involves building and refining prototypes.",
                    "no": "Less frequent requirement changes might make the Prototype model less necessary, as building and refining prototypes might not be as valuable."
                },
                "Iterative": {
                    "no": "With fewer requirement changes, the Iterative model might be more suitable as it allows the project to progress through iterations, refining requirements along the way.",
                    "yes": "Frequent requirement changes might make the Iterative model less suitable, as adapting to changes can be more challenging in an iterative process."
                },
                "Spiral": {
                    "yes": "Frequent requirement changes can be managed with the Spiral model as its risk-driven approach helps to manage uncertainty and adapt to changing requirements.",
                    "no": "Less frequent requirement changes might make the Spiral model less necessary, as its focus on managing risk and adapting to changes might not provide significant benefits."
                },
                "RAD": {
                    "no": "With fewer requirement changes, the RAD model might be more suitable as it relies on rapid application development and quick delivery, which can be hindered by frequent requirement changes.",
                    "yes": "Frequent requirement changes might make the RAD model less suitable, as adapting to changes can be more challenging in a rapid development process."
                },
                "XP": {
                    "yes": "Frequent requirement changes can be managed with the XP model as it emphasizes continuous integration and quick feedback, allowing for adaptation and clarification of requirements.",
                    "no": "Less frequent requirement changes might make the XP model less necessary, as its focus on continuous integration and quick feedback might not provide significant benefits."
                }
            },
            "Requirements are defined early in the cycle?": {
                "Waterfall": {
                    "yes": "This is good for the Waterfall model as it requires all the requirements to be defined at the beginning of the project.",
                    "no": "This is not good for the Waterfall model as it heavily relies on having all the requirements defined at the beginning of the project."
                },
                "Prototype": {
                    "no": "This is good for the Prototype model as it is designed to handle projects with evolving requirements by building and refining prototypes.",
                    "yes": "This is not as advantageous for the Prototype model as its strength lies in handling evolving requirements through iterative prototyping."
                },
                "Iterative": {
                    "yes": "This is good for the Iterative model as it allows for a more streamlined development process with well-defined requirements from the start.",
                    "no": "This is not as advantageous for the Iterative model as it is designed to handle changes in requirements throughout the project."
                },
                "Spiral": {
                    "no": "This is good for the Spiral model as it is designed to handle projects with uncertain requirements by incorporating risk analysis and iterative development.",
                    "yes": "This is not as advantageous for the Spiral model as its strength lies in handling uncertain requirements through risk-driven iterations."
                },
                "RAD": {
                    "yes": "This is good for the RAD model as it allows for a more streamlined development process with well-defined requirements from the start.",
                    "no": "This is not as advantageous for the RAD model as it is designed to handle rapid development, which can be hindered by uncertainty in requirements."
                },
                "XP": {
                    "no": "This is good for the XP model as it is designed to handle projects with evolving requirements by emphasizing continuous integration and quick feedback.",
                    "yes": "This is not as advantageous for the XP model as its strength lies in handling evolving requirements through continuous integration and quick feedback."
                }
            },
            "Is the system complex due to the requirements?": {
                "Waterfall": {
                    "no": "This is good for the Waterfall model as it works better with projects that have simple and well-defined requirements.",
                    "yes": "This is not good for the Waterfall model as it may struggle to handle projects with complex and evolving requirements."
                },
                "Prototype": {
                    "yes": "This is good for the Prototype model as it is designed to handle complex systems by building and refining prototypes to address evolving requirements.",
                    "no": "This is not as advantageous for the Prototype model as its strength lies in handling complex systems with evolving requirements."
                },
                "Iterative": {
                    "yes": "This is good for the Iterative model as it is designed to handle complex systems by breaking them down into smaller, manageable iterations.",
                    "no": "This is not as advantageous for the Iterative model as its strength lies in handling complex systems through iterative development."
                },
                "Spiral": {
                    "yes": "This is good for the Spiral model as it is designed to handle complex systems by incorporating risk analysis and iterative development.",
                    "no": "This is not as advantageous for the Spiral model as its strength lies in handling complex systems through risk-driven iterations."
                },
                "RAD": {
                    "no": "This is good for the RAD model as it works better with projects that have simple and well-defined requirements.",
                    "yes": "This is not good for the RAD model as it may struggle to handle projects with complex and evolving requirements."
                },
                "XP": {
                    "yes": "This is good for the XP model as it is designed to handle complex systems by emphasizing continuous integration and quick feedback.",
                    "no": "This is not as advantageous for the XP model as its strength lies in handling complex systems through continuous integration and quick feedback."
                }
            },
            "The users participate in all the phases?": {
                "Waterfall": {
                    "no": "This is good for the Waterfall model as it doesn't require user involvement throughout all phases, with users mainly involved in the requirements phase.",
                    "yes": "This is not as advantageous for the Waterfall model as it follows a more rigid structure, with less need for user involvement throughout all phases."
                },
                "Prototype": {
                    "yes": "This is good for the Prototype model as user feedback throughout all phases helps in refining the prototypes and improving the final solution.",
                    "no": "This is not as advantageous for the Prototype model as its strength lies in utilizing user feedback to refine prototypes throughout the development process."
                },
                "Iterative": {
                    "no": "This is good for the Iterative model as it allows for some user involvement, but it doesn't necessarily require user participation in all phases.",
                    "yes": "This is not as advantageous for the Iterative model as it doesn't strictly require user involvement throughout all phases of the project."
                },
                "Spiral": {
                    "no": "This is good for the Spiral model as it doesn't require user involvement throughout all phases, with users mainly involved in the evaluation and risk analysis stages.",
                    "yes": "This is not as advantageous for the Spiral model as it follows a more risk-driven approach, with less need for user involvement throughout all phases."
                },
                "RAD": {
                    "yes": "This is good for the RAD model as it emphasizes user involvement throughout all phases to ensure the final solution meets user needs.",
                    "no": "This is not as advantageous for the RAD model as its strength lies in utilizing user feedback to continuously improve the solution."
                },
                "XP": {
                    "yes": "This is good for the XP model as it encourages user involvement throughout all phases to ensure the final solution is aligned with user needs.",
                    "no": "This is not as advantageous for the XP model as its strength lies in incorporating user feedback throughout the development process."
                }
            },
            "Is there limited user participation?": {
                "Waterfall": {
                    "yes": "This is good for the Waterfall model as it doesn't require extensive user involvement, with users mainly participating in the requirements phase.",
                    "no": "This is not as advantageous for the Waterfall model as it doesn't benefit significantly from extensive user involvement throughout all phases."
                },
                "Prototype": {
                    "no": "This is good for the Prototype model as it relies on user feedback throughout the development process to refine and improve the prototypes.",
                    "yes": "This is not as advantageous for the Prototype model as limited user participation may hinder the refinement and improvement of prototypes."
                },
                "Iterative": {
                    "yes": "This is good for the Iterative model as it allows for some user involvement but doesn't necessarily require extensive user participation.",
                    "no": "This is not as advantageous for the Iterative model as extensive user participation isn't strictly required throughout all phases of the project."
                },
                "Spiral": {
                    "yes": "This is good for the Spiral model as it doesn't require extensive user involvement throughout all phases, with users mainly participating in the evaluation and risk analysis stages.",
                    "no": "This is not as advantageous for the Spiral model as it doesn't benefit significantly from extensive user involvement throughout all phases."
                },
                "RAD": {
                    "no": "This is good for the RAD model as it emphasizes user involvement throughout all phases to ensure the final solution meets user needs.",
                    "yes": "This is not as advantageous for the RAD model as limited user participation may hinder the continuous improvement of the solution."
                },
                "XP": {
                    "no": "This is good for the XP model as it encourages user involvement throughout all phases to ensure the final solution is aligned with user needs.",
                    "yes": "This is not as advantageous for the XP model as limited user participation may hinder the incorporation of user feedback throughout the development process."
                }
            },
            "Does the user have any experience on similar projects?": {
                "Waterfall": {
                    "no": "This is good for the Waterfall model as it doesn't rely heavily on user expertise, focusing more on a structured, linear process.",
                    "yes": "This is not as advantageous for the Waterfall model as user experience doesn't play a critical role in the development process."
                },
                "Prototype": {
                    "yes": "This is good for the Prototype model as user experience on similar projects can help in providing valuable feedback and shaping the prototypes.",
                    "no": "This is not as advantageous for the Prototype model as the lack of user experience may affect the quality of feedback during the prototype refinement process."
                },
                "Iterative": {
                    "yes": "This is good for the Iterative model as user experience on similar projects can contribute to better feedback and insights during each iteration.",
                    "no": "This is not as advantageous for the Iterative model as the lack of user experience may hinder the improvement of the solution during each iteration."
                },
                "Spiral": {
                    "yes": "This is good for the Spiral model as user experience on similar projects can contribute to better risk analysis and evaluation during the development process.",
                    "no": "This is not as advantageous for the Spiral model as the lack of user experience may affect the quality of risk analysis and evaluation."
                },
                "RAD": {
                    "no": "This is good for the RAD model as it focuses on rapid development and doesn't heavily rely on user expertise in similar projects.",
                    "yes": "This is not as advantageous for the RAD model as user experience doesn't play a critical role in the rapid development process."
                },
                "XP": {
                    "no": "This is good for the XP model as it emphasizes flexibility and continuous improvement, without heavily relying on user expertise in similar projects.",
                    "yes": "This is not as advantageous for the XP model as user experience doesn't play a critical role in the flexible and iterative development process."
                }
            },
            "Are the users experts of the problem domain?": {
                "Waterfall": {
                    "no": "This is good for the Waterfall model as it follows a structured, linear process and doesn't heavily rely on user expertise in the problem domain.",
                    "yes": "This is not as advantageous for the Waterfall model as user expertise in the problem domain doesn't play a critical role in the development process."
                },
                "Prototype": {
                    "yes": "This is good for the Prototype model as users with expertise in the problem domain can provide valuable feedback to improve and refine the prototypes.",
                    "no": "This is not as advantageous for the Prototype model as the lack of user expertise in the problem domain may affect the quality of feedback during the prototype refinement process."
                },
                "Iterative": {
                    "yes": "This is good for the Iterative model as user expertise in the problem domain can contribute to better feedback and insights during each iteration.",
                    "no": "This is not as advantageous for the Iterative model as the lack of user expertise in the problem domain may hinder the improvement of the solution during each iteration."
                },
                "Spiral": {
                    "no": "This is good for the Spiral model as it focuses on risk analysis and evaluation and doesn't heavily rely on user expertise in the problem domain.",
                    "yes": "This is not as advantageous for the Spiral model as user expertise in the problem domain doesn't play a critical role in the risk analysis and evaluation process."
                },
                "RAD": {
                    "yes": "This is good for the RAD model as users with expertise in the problem domain can help in shaping the rapid development and providing useful feedback.",
                    "no": "This is not as advantageous for the RAD model as the lack of user expertise in the problem domain may affect the quality of feedback and rapid development."
                },
                "XP": {
                    "yes": "This is good for the XP model as users with expertise in the problem domain can provide valuable insights and contribute to the continuous improvement process.",
                    "no": "This is not as advantageous for the XP model as the lack of user expertise in the problem domain may affect the quality of feedback and continuous improvement."
                }
            },
        }
        
        user_responses = []
        for label in data['results']:
            for answer in label['answers']:
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
        explanations = {}
        for sdlc, count in scores.items():
            percentage = count / 18 * 100
            percentages[sdlc] = round(percentage, 1)


            explanations[sdlc] = ""

            for idx, question in enumerate(questions["Risk"]):
                if question in explanation_map:
                    sdlc_explanation = explanation_map[question][sdlc]
                    user_response = user_responses[idx]
                    explanations[sdlc] += sdlc_explanation[user_response] + " "

        print("Explanations:")
        print(explanations)
        # sort the SDLCs by their percentage match
        sorted_sdlcs = sorted(percentages.items(), key=lambda x: x[1], reverse=True)

        final_output = [{"sdlc": sdlc, "percentage": percentage, "explanation": explanations[sdlc]} for sdlc, percentage in sorted_sdlcs]

        return jsonify({'sorted_SDLCs': final_output})
    except Exception as e:
        print("Exception occurred:", e)
        raise



if __name__ == '__main__':
    app.run()
