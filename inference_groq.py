import os
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm
import json
import random
from sklearn.metrics import f1_score

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = Groq(
    # This is the default and can be omitted
    api_key=api_key,
)
with open('ntu-nlp-2024/team_dev.json', encoding='utf-8') as json_file:
    dev_data = json.load(json_file)

table = [[0 for i in range(3)] for j in range(3)]
random.shuffle(dev_data)
data = dev_data[132]
predicts = []
golden = []
# for data in tqdm(dev_data[:50]):
for data in tqdm(dev_data):
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are an expert in finance, social media and Mandarin. Determine the relationship between the following post and comment about finance. Does the comment support, attack, or do nothing to the post? You need to give the answer in the form \"{answer}\", where {answer} can only be \'support\', \'attack\', or \'none\'. There is no need for explanation.\n\
            \n\
            Post: \"早上買入兩張建立持股  現階段的電信股除了威寶和亞太之外  三雄的使用人數都還在上升(吸納了亞太電信數十萬的用戶)  長期來說沒有悲觀的理由\"\n\
            Comment: \"下殺是帶量的  或許會回測前波低點87.2\"\n\
            Classification: support\n\
            \n\
            Post: \"這兩天美股大跌 友達ADR也是一樣 10/10 友達ADR 收盤價已跌到 美金$3.95 相當於台幣12元不到 不知道 今天的友達是否會反應這個ADR的股價。\"\n\
            Comment: \"漲停如果還不算漲幅很大,那要怎樣才算漲幅很大?\"\n\
            Classification: attack\n\
            \n\
            Post: \"看看憐發科這波會不會跌破前波低點193.5.\"\n\
            Comment: \"發哥年底也有十奈米的晶片要上市啊\n 為什麼股價又見到一字頭了。\"\n\
            Classification: none"
        },
        {
            "role": "user",
            "content": f"Post: \"{data[1]}\"\n\
            Comment: \"{data[2]}\"\n\
            Classification: ",
        }
    ],
    model="llama3-70b-8192",
    max_tokens=20,
    )

    answer = chat_completion.choices[0].message.content
    if 'none' in answer:
        predict = 0
    elif 'support' in answer:
        predict = 1
    elif 'attack' in answer:
        predict = 2
    else:
        predict = random.randint(0,2)
    predicts.append(predict)
    golden.append(data[-1])
    table[predict][data[-1]] += 1

print("F1_score:", f1_score(golden, predicts, average='weighted'))
print(table)