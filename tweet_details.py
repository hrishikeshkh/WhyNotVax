import csv
import http.client
import json
import pandas as pd
import numpy as np

data_frame = pd.read_csv('Book1.csv')
number_column = data_frame['ID']
numpy_array = np.array(number_column)
modified_array = np.array([value[:-1] for value in numpy_array])
tweet_ids = modified_array.astype(str).tolist()

def get_tweet_details(tweet_id):
    conn = http.client.HTTPSConnection("twitter154.p.rapidapi.com")

    headers = {
        'X-RapidAPI-Key': "3d5767e85amshec06c0701d96ffap19849bjsne43ef91fe049",
        'X-RapidAPI-Host': "twitter154.p.rapidapi.com"
    }

    conn.request("GET", f"/tweet/details?tweet_id={tweet_id}", headers=headers)

    res = conn.getresponse()
    data = res.read()

    return data.decode("utf-8")

# tweet_ids = ["1413367208537989123", "1413367208537989123"]

fields = [
    'tweet_id',
    'creation_date',
    'text',
    'user.username',
    'user.name',
    'user.follower_count',
    'user.following_count',
    'user.number_of_tweets',
    'favorite_count',
    'retweet_count',
    'reply_count',
    'quote_count',
    'retweet',
    'source'
]

# Create a new CSV file
with open("tweet_details.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)

    # Write the header row
    writer.writerow(fields)
    
    # Fetch tweet details and write to the CSV file
    limit = 10
    for index, tweet_id in enumerate(tweet_ids):
        if index >= limit:
            break
        tweet_data = get_tweet_details(tweet_id)
        tweet_dict = json.loads(tweet_data)

        # Write the data rows
        row = []
        for field in fields:
            keys = field.split('.')
            value = tweet_dict
            for key in keys:
                value = value.get(key)
                if value is None:
                    break
            row.append(value)
        writer.writerow(row)
print("CSV file created successfully.")
