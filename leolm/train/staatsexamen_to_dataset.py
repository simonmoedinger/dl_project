import pandas as pd

prompt_case = "Es folgt eine Fallbeschreibung, dessen Anfang und Ende durch die Tags BEGINFALL und ENDFALL markiert ist. Darauf wird eine Frage gestellt, die zwischen den TAGS BEGINFRAGE und ENDFRAGE steht. Die Antwortmöglichkeiten stehen jeweils zwischen den Tags BEGINANTWORT und ENDANTWORT. BEGINFALL <INSERTFALL> ENDFALL BEGINFRAGE <INSERTFRAGE> ENDFRAGE BEGINANTWORT <INSERTANTWORT0> ENDANTWORT BEGINANTWORT <INSERTANTWORT1> ENDANTWORT BEGINANTWORT <INSERTANTWORT2> ENDANTWORT BEGINANTWORT <INSERTANTWORT3> ENDANTWORT BEGINANTWORT <INSERTANTWORT4> ENDANTWORT. Beantworte die Frage, indem du die korrekte Antwortmöglichkeit wiedergibst."
prompt_no_case = "Es wird eine Frage gestellt, die zwischen den TAGS BEGINFRAGE und ENDFRAGE steht. Die Antwortmöglichkeiten stehen jeweils zwischen den Tags BEGINANTWORT und ENDANTWORT. BEGINFALL <INSERTFALL> ENDFALL BEGINFRAGE <INSERTFRAGE> ENDFRAGE BEGINANTWORT <INSERTANTWORT0> ENDANTWORT BEGINANTWORT <INSERTANTWORT1> ENDANTWORT BEGINANTWORT <INSERTANTWORT2> ENDANTWORT BEGINANTWORT <INSERTANTWORT3> ENDANTWORT BEGINANTWORT <INSERTANTWORT4> ENDANTWORT. Beantworte die Frage, indem du die korrekte Antwortmöglichkeit wiedergibst."


dataset = open("instruct_dataset/instructs.txt", "a")

questions_df = pd.read_csv("../../staatsexamen/hs1_hs2.csv")

for index, row in questions_df.iterrows():
    if type(row["Case"]) == str:
        prompt = prompt_case.replace("<INSERTFALL>", row["Case"])
    else:
        prompt = prompt_no_case
    prompt = prompt.replace("<INSERTFRAGE>", row["Question"])
    for i in range(5):
        prompt = prompt.replace("<INSERTANTWORT" + str(i) + ">", row["Answer " + str(i + 1)].replace(str(i) + ")", ""))
    prompt = prompt.replace('\n','')
    answer = row["Answer " + str(int(row["Correct Answer"]) + 1)].replace('\n','')
    dataset.write(f"Du bist ein hilfreicher Assistent, der alle Fragen so korrekt wie möglich beantwortet. <BENUTZER>: {prompt} <ASSISTENT>: BEGINANTWORT {answer} ENDANTWORT\n")

dataset.close()