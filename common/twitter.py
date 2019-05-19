import twitter
import os
import yaml

curDir = os.path.dirname(__file__)
creds = os.path.join(curDir, "../data/twitter_keys.yaml")
with open(os.path.expanduser(creds)) as f:
    params = yaml.load(f)['search_tweets_api']
api = twitter.Api(**params)

def search(query, count=15):
    result = api.GetSearch(term=query, count=count)
    return result

def getById(id):
    result = api.GetStatus(id)
    return result

def getByIds(ids):
    result = api.GetStatuses(ids)
    return result