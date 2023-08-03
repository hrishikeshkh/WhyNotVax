import pandas as pd
import re

def clean(text):
    # Remove "@" tags
    text = re.sub(r'@\w+', '', text)
    
    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove URLs
    text = re.sub(r'https?:\/\/\S+', '', text)
    
    # Remove special characters
    special_char_pattern = re.compile(r'[^a-zA-Z0-9\s]')
    text = special_char_pattern.sub('', text)

    return text

# Read the CSV file
input_file = 'data/train_val.csv'
output_file = 'data/train_val_cleaned.csv'

df = pd.read_csv(input_file)

# Clean the 'tweet' column
df['tweet'] = df['tweet'].apply(clean)

# Save the cleaned data to a new CSV file
df.to_csv(output_file, index=False)

print("Data cleaned and saved to:", output_file)
