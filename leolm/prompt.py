import math

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, GenerationConfig, BitsAndBytesConfig
import torch
from tqdm import tqdm

#model_name = "fully_trained_model/"
model_name = "LeoLM/leo-mistral-hessianai-7b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = False
tokenizer.add_bos_token, tokenizer.add_eos_token

generation_config = GenerationConfig(max_new_tokens=100,
                                    temperature=0.4,
                                    top_p=0.95,
                                    top_k=40,
                                    repetition_penalty=1.3,
                                    bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    do_sample=True,
                                    use_cache=True,
                                    output_attentions=False,
                                    output_hidden_states=False,
                                    output_scores=False,
                                    remove_invalid_values=True
                                    )

def prompt_model(prompt):
    input_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_tokens = model.generate(**input_tokens, generation_config=generation_config, pad_token_id=tokenizer.bos_token_id, streamer=TextStreamer(tokenizer))[0]
    answer = tokenizer.decode(output_tokens, skip_special_tokens=True)
    return answer

prompt_model(
        """ Es folgt eine Fallbeschreibung, dessen Anfang und Ende durch die Tags BEGINFALL und ENDFALL markiert ist. Darauf wird eine Frage gestellt, die zwischen den TAGS BEGINFRAGE und ENDFRAGE steht. Die Antwortmöglichkeiten stehen jeweils zwischen den Tags BEGINANTWORT und ENDANTWORT. BEGINFALL Eine 36-jährige Sekretärin stellt sich mit Fieber von etwa 38,5 °C sowie reichlich gelblichem Auswurf und Abgeschlagenheit in der Hausarztpraxis vor. Außer einer Appendektomie vor ca. 1 ½ Jahren sind keine Vorerkrankungen und keine Voroperationen bekannt.<br> Die Patientin ist verheiratet und hat 2 Kinder im Vorschulalter. Befunde der körperlichen Untersuchung:<br> Patientin in gutem AZ und EZ, vollorientiert. Auskultatorisch bestehen ein Bronchialatmen und klingende Atemgeräusche rechts im Mittelfeld, keine pathologischen Nebengeräusche; leise Herztöne. RR 110/70 mmHg, HF 96/min, AF 19/min. ENDFALL BEGINFRAGE Welche der genannten Untersuchungen gilt am ehesten als Standard zur Bestätigung der Verdachtsdiagnose? ENDFRAGE BEGINANTWORT  Spirometrie ENDANTWORT BEGINANTWORT  Blutgasanalyse ENDANTWORT BEGINANTWORT  Röntgenaufnahme des Thorax ENDANTWORT BEGINANTWORT  Sonografie des Thorax ENDANTWORT BEGINANTWORT  Blutbilduntersuchung ENDANTWORT. Beantworte die Frage, indem du die korrekte Antwortmöglichkeit wiedergibst. Die richtige Antwort ist: BEGINANTWORT 2) Röntgenaufnahme des Thorax ENDANTWORT.
Es folgt eine Fallbeschreibung, dessen Anfang und Ende durch die Tags BEGINFALL und ENDFALL markiert ist. Darauf wird eine Frage gestellt, die zwischen den TAGS BEGINFRAGE und ENDFRAGE steht. Die Antwortmöglichkeiten stehen jeweils zwischen den Tags BEGINANTWORT und ENDANTWORT. BEGINFALL Ihr Patient ist 76 Jahre alt und bedarf wegen permanenten Vorhofflimmerns mit einem CHA<sub><small>2</small></sub>DS<sub><small>2</small></sub>-VASc-Score von 3 einer oralen Antikoagulation (OAK), die Sie mit Rivaroxaban 20 mg 1 x tgl. eingeleitet haben. Einzige relevante Begleiterkrankung ist eine Hypertonie (aktuell mit Amlodipin und Enalapril/Hydrochlorothiazid behandelt). Die Nierenfunktion ist gut (eGFR nach CKD-EPI-Formel 65 mL/min). ENDFALL BEGINFRAGE Direkt wirkende orale Antikoagulanzien können neben der Anwendung bei nichtvalvulärem Vorhofflimmern auch für andere Indikationen eingesetzt werden.  Welches der folgenden Anwendungsgebiete gehört dazu? ENDFRAGE BEGINANTWORT  Behandlung der tiefen Venenthrombose und Lungenembolie ENDANTWORT BEGINANTWORT  Antikoagulation bei künstlichem Herzklappenersatz ENDANTWORT BEGINANTWORT  Therapie der disseminierten Verbrauchskoagulopathie ENDANTWORT BEGINANTWORT  Gerinnungshemmung bei extrakorporalem Kreislauf während der Hämodialyse ENDANTWORT BEGINANTWORT  Therapie des akuten ST-Hebungs-Myokardinfarktes ENDANTWORT. Beantworte die Frage, indem du die korrekte Antwortmöglichkeit wiedergibst. Die richtige Antwort ist: BEGINANTWORT 0) Behandlung der tiefen Venenthrombose und Lungenembolie ENDANTWORT.
Es folgt eine Fallbeschreibung, dessen Anfang und Ende durch die Tags BEGINFALL und ENDFALL markiert ist. Darauf wird eine Frage gestellt, die zwischen den TAGS BEGINFRAGE und ENDFRAGE steht. Die Antwortmöglichkeiten stehen jeweils zwischen den Tags BEGINANTWORT und ENDANTWORT. BEGINFALL Wegen postprandialer rechtsseitiger Oberbauchbeschwerden wird bei einer 59-jährigen adipösen Frau (BMI 30,1 kg/m<sup><small>2</small></sup>) eine Abdomensonografie veranlasst. Gallensteine oder andere pathologische Befunde der Gallenblase, der Gallenwege oder der Leber zeigen sich bei dieser Untersuchung nicht, jedoch findet sich im Bereich des Pankreasschwanzes als Zufallsbefund eine echoarme Raumforderung mit einem Durchmesser von 1,6 cm. In der Kontrastmittel-unterstützten CT des Abdomens wird diese als glatt begrenzter, arteriell verstärkt perfundierter Tumor beschrieben, der übrige radiologische Befund ist unauffällig. Nach Abschluss der Diagnostik wird in der interdisziplinären Tumorkonferenz letztlich die Indikation zur Pankreasteilresektion gestellt. Die operative Therapie verläuft insgesamt ohne Besonderheiten. Histologisch wird ein hochdifferenzierter neuroendokriner Tumor des Pankreas (NET G1) mit vollständiger Resektion (R0) diagnostiziert. ENDFALL BEGINFRAGE Im Rahmen der Pankreaslinksresektion wurde bei der Patientin aus technischen Gründen auch eine Splenektomie durchgeführt. Aus diesem speziellen Grund ist bei ihr (gemäß STIKO) eine Impfmaßnahme indiziert.  Gegen welche(n) der genannten Erreger richtet sich die erforderliche Impfmaßnahme vorrangig? ENDFRAGE BEGINANTWORT 0) Hepatitis-A-Virus ENDANTWORT BEGINANTWORT 1) Mycobacterium tuberculosis ENDANTWORT BEGINANTWORT 2) Varicella-Zoster-Virus ENDANTWORT BEGINANTWORT 3) Pneumokokken, Meningokokken und Haemophilus influenzae Typ b ENDANTWORT BEGINANTWORT 4) Masern-Virus, Mumps-Virus und Bordetella pertussis ENDANTWORT. Beantworte die Frage, indem du die korrekte Antwortmöglichkeit wiedergibst. Die richtige Antwort ist: BEGINANTWORT""")