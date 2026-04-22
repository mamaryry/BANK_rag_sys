import time

CACHE = {}

def timed_response(fn, query):
    start = time.time()
    result = fn(query)
    end = time.time()
    return result, end - start

def cached_query(query, fn):
    if query in CACHE:
        return CACHE[query], True

    result = fn(query)
    CACHE[query] = result
    return result, False