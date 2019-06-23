import redis
import pickle
import hashlib
import functools

pool = redis.ConnectionPool(host='176.223.134.183', port=14890, password='vr@`~ud`b#U5{}v=~z*BANPJW>WAA-dJ!+D:5{r:sA,=[[.bWpLRQX#;Y~bp', db=0)
r = redis.Redis(connection_pool=pool)


def generateIdentifier(text, strict=False):
    """
    генерация идентификатора для текстового значения
    :param text: значене, для которого нужно определить идентификатор
    :type text: str
    :param strict: осуществлять ли более строгую (и более ресурсозатратную) генерацию идентификатора
    :type strict: bool
    :return: идентификатор
    :rtype: str
    """
    hashString = hashlib.md5(text.encode("utf-8")).hexdigest()
    if strict:
        hashString = hashString + hashlib.sha1(text.encode("utf-8")).hexdigest()
    return hashString


def save(name, value):
    """
    сохранение значения value в кеше с идентификатором name
    :param name: идентификатор сохраняемого значения
    :type name: str
    :param value: сохраняемое значение
    :type value: str
    :return: результат сохранения
    :rtype: bool
    """
    return saveByte(name, value.encode('utf-8'))


def read(name):
    """
    чтение значения с идентификатором name
    :param name: идентификатор сохраняемого значения
    :type name: str
    :return: результат чтения
    :rtype: str
    """
    return readByte(name).decode('utf-8')


def saveByte(name, value):
    """
    запись байтового значения value в кеше с идентификатором name
    :param name: идентификатор сохраняемого значения
    :type name: str
    :param value: сохраняемое значение
    :type value: bytes
    :return: результат сохранения
    :rtype: bool
    """
    try:
        result = r.set(name, value)
    except redis.exceptions.ConnectionError:
        result = None
    return result


def readByte(name):
    """
    чтение байтового значения с идентификатором name
    :param name: идентификатор сохраняемого значения
    :type name: str
    :return: результат чтения
    :rtype: bytes
    """
    try:
        result = r.get(name)
        if result is not None:
            result = result
    except redis.exceptions.ConnectionError:
        result = None
    return result


def saveVar(name, value):
    """
    сохранение переменной в кеш
    :param name: название (идентификатор) переменной
    :type name: str
    :param value: сохраняемая переменная
    :type value: object
    :return: object
    :rtype: bool
    """
    rawValue = pickle.dumps(value)
    return saveByte(name, rawValue)


def readVar(name):
    """
    чтение переменной из кеша
    :param name: название (идентификатор) переменной
    :type name: str
    :return: переменная из кеша
    :rtype: object
    """
    result = None
    rawResult = readByte(name)
    if rawResult is not None:
        result = pickle.loads(rawResult)
    return result



def cacheFuncOutput(func):
    """обёртка для реализации кеширования выхода функции"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        if 'cacheHash' in kwargs and kwargs['cacheHash'] is not None:
            # если есть хеш, пытаемся прочитать по нему из кеша
            result = readVar(kwargs['cacheHash'])
            if result is not None:
                return result
        value = func(*args, **kwargs)
        if 'cacheHash' in kwargs and kwargs['cacheHash'] is not None:
            # если есть хеш, пытаемся сохранить по нему
            saveVar(kwargs['cacheHash'], value)
        return value
    return wrapper_timer