import json
import os
import random
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file')
parser.add_argument('-t', '--type')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

with open(args.file) as json_file:
    train_data = json.load(json_file)

type_map = ["none", "support", "attack"]
random.shuffle(train_data)
final_data = []
system_prompt = "Determine the relationship between the following post and comment about finance. Does the comment support, attack, or do nothing to the post? You need to give the answer in the form \"Relationship: {answer}\", where {answer} can only be \'support\', \'attack\', or \'none\'. There is no need for explanation."
with open(f'ntu-nlp-2024/preprocessed_{args.type}.json', '+w', encoding='UTF-8') as preprocessed_file:
    for data in tqdm(train_data):
        line_data = {'instruction': system_prompt}
        comment = [data[1], data[2]]
        user_msg = f"\"{comment[0]}\", \"{comment[1]}\""
        line_data['input'] = user_msg
        if(not args.test):
            model_answer = f'Relationship: {type_map[data[3]]}'
            line_data['output'] = model_answer      
        final_data.append(line_data)
    json_data = json.dumps(final_data, indent=4, ensure_ascii=False)
    preprocessed_file.write(json_data)
