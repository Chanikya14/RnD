# services/memcached.py
class Memcached:
    """Simulated Memcached Cache"""
    def __init__(self):
        self.timeline_cache = {}

    def cache_timeline(self, user, post_id):
        """Cache user timeline in Memcached"""
        self.timeline_cache[user] = post_id
        return f"⚡ Memcached: Timeline updated for {user} with {post_id}"

    def read_timeline(self, user):
        """Retrieve cached timeline for user"""
        return self.timeline_cache.get(user, None)
