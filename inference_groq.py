import os
from dotenv import load_dotenv
from groq import Groq
from together import Together
from tqdm import tqdm
import json
import random
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")

with open('ntu-nlp-2024/team_dev.json', encoding='utf-8') as json_file:
    dev_data = json.load(json_file)

with open('ntu-nlp-2024/team_train.json', encoding='utf-8') as json_file:
    train_data = json.load(json_file)
    
table = [[0 for i in range(3)] for j in range(3)]
random.shuffle(dev_data)
data_none = list(filter(lambda x: x[-1] == 0 , train_data))
data_support = list(filter(lambda x: x[-1] == 1 , train_data))
data_attack = list(filter(lambda x: x[-1] == 2 , train_data))
predicts = []
golden = []
relationship = ["無關", "支持", "反對"]
system_prompt = '任務說明：你是一位來自台灣的金融及股票專家，並且熟知網路用語及繁體中文中關於台灣的金融、股票的一些專有名詞。以下是一些關於台灣金融話題的繁體中文貼文和評論。請仔細閱讀每個貼文和相應的評論，然後分類最後一則貼文和相應的評論之間的關係。可能的分類包括："支持"、"反對"、"無關"。請以"關係：{分類}"的形式回答，包括你所判斷的關係分類，你只需要回答你的分類即可。請確保您的回答是基於內容的分析，並且以繁體中文呈現。'
print(len(data_none))
print(len(data_support))
print(len(data_attack))
print(system_prompt)
# data_test = random.choice(data_none)
context_template = '貼文："{post}"。\n\
評論："{comment}"。\n\
關係："{relationship}"'
# context_content = context_template.format(post = data_test[1], comment = data_test[2], relationship = relationship[data_test[3]])
# print(context_content)
# exit()

# client_groq = Groq(
#     # This is the default and can be omitted
#     api_key=groq_api_key,
# )

client_together = Together(
    api_key=together_api_key
)

for data in tqdm(dev_data):
    run = False
    context_none = random.choices(data_none, k=6)
    context_support = random.choices(data_support, k=3)
    context_attack = random.choices(data_attack, k=6)
    context = context_none + context_support + context_attack
    context_content = map(lambda x: context_template.format(post = x[1], comment = x[2], relationship = relationship[x[3]]), context)
    context_content = '\n\n'.join(context_content)
    # print(context_content)
    question = f'貼文："{data[1]}"。\n評論："{data[2]}"\n關係：。'
    # print(f"{context_content}\n\n{question}")
    while(True):
        try:
            ## groq
            
            # chat_completion = client_groq.chat.completions.create(
            # messages=[
            #     {
            #         "role": "system",
            #         "content": f"{system_prompt}"
            #     },
            #     {
            #         "role": "user",
            #         "content": f"{context_content}\n\n{question}",
            #     }
            # ],
            # model="llama3-70b-8192",
            # max_tokens=20,
            # temperature=0.75,
            # top_p=0.9
            # )
            
            ## together
            
            chat_completion = client_together.chat.completions.create(
                model="meta-llama/Llama-3-70b-chat-hf",
                messages=[
                    {
                        "role": "system",
                        "content": f"{system_prompt}"
                    },
                    {
                        "role": "user",
                        "content": f"{context_content}\n\n{question}"
                    }
                ],
                top_p=0.9,
                temperature=0.75,
                max_tokens=20,
            )
            break
        except KeyboardInterrupt:
            print("Program stopped.")
            exit()
        except Exception:
            pass
    
    answer = chat_completion.choices[0].message.content
    # print(answer)
    if '無關' in answer:
        predict = 0
    elif '支持' in answer:
        predict = 1
    elif '反對' in answer:
        predict = 2
    else:
        predict = random.randint(0,2)
    predicts.append(predict)
    golden.append(data[-1])
    table[predict][data[-1]] += 1

print("F1_score:", f1_score(golden, predicts, average='weighted'))
cm = confusion_matrix(y_true=golden, y_pred=predicts)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
disp.plot()
plt.show()