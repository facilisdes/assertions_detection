import twitter
import os
import yaml

curDir = os.path.dirname(__file__)
creds = os.path.join(curDir, "../data/twitter_keys.yaml")
with open(os.path.expanduser(creds)) as f:
    params = yaml.load(f)['search_tweets_api']
api = twitter.Api(**params)


def search(query, count=15):
    """
    поиск твитов по заданному запросу
    :param query: поисковой запрос
    :type query: str
    :param count: максимальное количество записей, которые нужно вернуть
    :type count: int
    :return: результат поиска
    :rtype: List[twitter.models.Status]
    """
    result = api.GetSearch(term=query, count=count)
    return result


def getById(tweetId):
    """
    поиск твита по его id
    :param tweetId: id твита
    :type tweetId: str
    :return: результат поиска
    :rtype: object
    """
    result = api.GetStatus(tweetId)
    return result


def getByIds(tweetIds):
    """
    поиск твитов по их id
    :param tweetIds: id твитов
    :type tweetIds: str
    :return: результат поиска
    :rtype: object
    """
    result = api.GetStatuses(tweetIds)
    return result
