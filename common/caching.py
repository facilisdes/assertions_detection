import redis
import pickle
import hashlib

pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
r = redis.Redis(connection_pool=pool)


def generateIdentifier(text, strict=False):
    hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    if strict:
        hash = hash + hashlib.sha1(text.encode("utf-8")).hexdigest()
    return hash

def save(name, value):
    """
    сохранение значения value в кеше с идентификатором name
    :param name: идентификатор сохраняемого значения
    :type name: string
    :param value: сохраняемое значение
    :type value: string
    :return: результат сохранения
    :rtype: bool
    """
    return saveByte(name, value.encode('utf-8'))


def read(name):
    """
    чтение значения с идентификатором name
    :param name: идентификатор сохраняемого значения
    :type name: string
    :return: результат чтения
    :rtype: string
    """
    return readByte(name).decode('utf-8')


def saveByte(name, value):
    """
    запись байтового значения value в кеше с идентификатором name
    :param name: идентификатор сохраняемого значения
    :type name: string
    :param value: сохраняемое значение
    :type value: bytearray
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
    :type name: string
    :return: результат чтения
    :rtype: bytearray
    """
    try:
        result = r.get(name)
        if result is not None:
            result = result
    except redis.exceptions.ConnectionError:
        result = None
    return result


def saveVar(name, value):
    rawValue = pickle.dumps(value)
    return saveByte(name, rawValue)


def readVar(name):
    result = None
    rawResult = readByte(name)
    if rawResult is not None:
        result = pickle.loads(rawResult)
    return result