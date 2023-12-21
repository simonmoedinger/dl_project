import pandas as pd
import re

def parse_content(content):
    cases = content.split("<Fall>")
    data = []
    for case in cases:
        case_questions = case.split("<Frage>")
        case = case_questions[0]
        for question_answers in case_questions[1:]:
            question, answers = question_answers.split("<0>")
            answer0, answers = answers.split("<1>")
            answer1, answers = answers.split("<2>")
            answer2, answers = answers.split("<3>")
            answer3, answers = answers.split("<4>")
            answer4, correct_answer = answers.split("<Antwort>")
            data.append([case.strip(), question.strip(), answer0.strip(), answer1.strip(), answer2.strip(), answer3.strip(), answer4.strip(), correct_answer.strip()])


    return data

def write_to_csv(data, output_file):
    df = pd.DataFrame(data, columns=['Case', 'Question', 'Answer 1', 'Answer 2', 'Answer 3', 'Answer 4', 'Answer 5', 'Correct Answer'])
    df.to_csv(output_file, index=False)

# Replace 'input.txt' with the path to your input file and 'output.xlsx' with your desired output Excel file name
input_files = ['hs2_bis_2012_fj.txt', "hs1_rest.txt"]
output_file = 'hs1_hs2.csv'

data = []

for input_file in input_files:
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
    data += parse_content(content)
write_to_csv(data, output_file)


