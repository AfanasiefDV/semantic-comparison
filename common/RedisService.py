import redis

class RedisService:
    def __init__(self):
        self.redis = redis.Redis(
            host='localhost',
            port=6379,
            charset="utf-8",
            decode_responses=True,
            password='simpleredispass'
        )

    def getKey(self, key: str) -> bool:
        return self.redis.get(key)

    def existsKey(self, key: str) -> str:
        return self.redis.exists(key)

    def setKey(self, key: str, value: str):
        self.redis.set(key, value)
