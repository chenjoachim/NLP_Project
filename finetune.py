import os
from tqdm import tqdm
from together import Together
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
wandb_api_key = os.getenv("WANDB_API_KEY")
fileID = "file-4527e8b1-ed3e-43b8-a473-b248f1afcf6d" 

client = Together(api_key=api_key)

# resp = client.files.upload(file="ntu-nlp-2024/preprocessed_train.jsonl") # uploads a file if not uploaded

resp = client.fine_tuning.create(
    training_file = fileID,
    model = 'meta-llama/Meta-Llama-3-8B-Instruct',
    n_epochs = 3,
    n_checkpoints = 3,
    batch_size = 16,
    learning_rate = 1e-5,
    wandb_api_key = wandb_api_key
)

print(resp)