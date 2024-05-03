import os
import json
import random
from tqdm import tqdm
from together import Together
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

with open('ntu-nlp-2024/team_dev.json') as json_file:
    dev_data = json.load(json_file)

with open('ntu-nlp-2024/team_train.json') as json_file:
    train_data = json.load(json_file)

train_none = list(filter(lambda data: data[-1] == 0, train_data))
train_support = list(filter(lambda data: data[-1] == 1, train_data))
train_attack = list(filter(lambda data: data[-1] == 2, train_data))
type_map = ["none", "support", "attack"]
correct = 0

random.shuffle(dev_data)
client = Together(api_key=api_key)
model = "lllion66666@gmail.com/Meta-Llama-3-8B-Instruct-2024-05-02-13-36-00"
# prompt = "What’s the relationship between the following two Chinese paragraphs about finance, support, attack, or none? You need to give the answer in the form \"Relationship: {answer}\". There is no need for explanation."
prompt = "Determine the relationship between the following post and comment about finance. Does the comment support, attack, or do nothing to the post? You need to give the answer in the form \"{answer}\", where {answer} can only be \'support\', \'attack\', or \'none\'. There is no need for explanation."
for dev in tqdm(dev_data[:2]):
    comment = [dev[1], dev[2]]
    # in_context_data = [random.choice(train_none)[1:],random.choice(train_none)[1:],random.choice(train_none)[1:],
    #                random.choice(train_support)[1:],random.choice(train_support)[1:],random.choice(train_support)[1:],
    #                random.choice(train_attack)[1:],random.choice(train_attack)[1:],random.choice(train_attack)[1:]]
    # random.shuffle(in_context_data)
    # in_context = [f'Post:\"{data[0]}\", Comment:\"{data[1]}\", Relationship: {type_map[data[2]]}' for data in in_context_data]
    # in_context = '\n'.join(in_context)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f'{prompt}\nPost:\"{comment[0]}\", Comment:\"{comment[1]}\"'}],
        top_k=50,
        top_p=0.7,
        temperature=0.7,
        max_tokens=10
    )
    answer = response.choices[0].message.content.lower()
    print(answer)
    if 'none' in answer:
        predict = 0
    elif 'support' in answer:
        predict = 1
    elif 'attack' in answer:
        predict = 2
    else:
        predict = random.randint(0,2)
    correct += int(dev[-1] == predict)

accuracy = correct/len(dev_data)
print("Accuracy: ", accuracy)

