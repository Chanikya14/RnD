import random

# ------------------------------
# Simulated Microservices
# ------------------------------

class MongoDB:
    """Simulated MongoDB Storage"""
    def __init__(self):
        self.posts = {}

    def store_post(self, post_id, content):
        """Store post persistently in MongoDB"""
        self.posts[post_id] = content
        return f"📝 MongoDB: Stored post {post_id}"

    def read_post(self, post_id):
        """Retrieve post from MongoDB"""
        return self.posts.get(post_id, "❌ Post not found")

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

class MediaService:
    """Handles media processing (images/videos)"""
    def process_media(self, post_id, media_type):
        """Process media attached to post"""
        return f"📷 Media Processing for {post_id}: {media_type}"

class UserTagService:
    """Handles user tagging in posts"""
    def handle_tags(self, post_id, tags):
        """Process user tags in a post"""
        return f"🏷️ Tags Added for {post_id}: {', '.join(tags)}"

class WriteTimelineService:
    """Broadcasts post to user’s followers"""
    def broadcast_to_followers(self, post_id, followers):
        """Broadcast the post to all followers"""
        return f"📣 Broadcasted post {post_id} to followers: {', '.join(followers)}"

class PostStorageService:
    """Handles post storage and retrieval"""
    def __init__(self, mongo_db):
        self.mongo_db = mongo_db

    def store_post(self, post_id, content):
        """Store post in MongoDB"""
        return self.mongo_db.store_post(post_id, content)

    def retrieve_post(self, post_id):
        """Retrieve post from MongoDB"""
        return self.mongo_db.read_post(post_id)

# ------------------------------
# Workflow Execution Functions
# ------------------------------
def high_engagement(post_id):
    """Simulate engagement check (for analytics or recommendation)"""
    return random.choice([True, False])

def compose_post(user, content, media=None, tags=[], followers=[]):
    """Post Creation Workflow with Conditional Execution"""
    # Initialize services
    mongo_db = MongoDB()
    memcached = Memcached()
    media_service = MediaService()
    user_tag_service = UserTagService()
    write_timeline_service = WriteTimelineService(mongo_db)
    post_storage_service = PostStorageService(mongo_db)

    post_id = f"POST_{random.randint(1000, 9999)}"
    print(f"📝 {user} is composing a post...")

    # Store post in MongoDB
    storage_message = post_storage_service.store_post(post_id, content)
    print(storage_message)

    # Conditional: Process media
    if media:
        media_message = media_service.process_media(post_id, media)
        print(media_message)

    # Conditional: Handle user tags
    if tags:
        tags_message = user_tag_service.handle_tags(post_id, tags)
        print(tags_message)

    # Update timeline cache
    cache_message = memcached.cache_timeline(user, post_id)
    print(cache_message)

    # Conditional: If engagement is high, update analytics & recommendation engine
    if high_engagement(post_id):
        print(f"📊 High Engagement! Updating Analytics and Recommendations for {post_id}")

    # Broadcast post to followers
    broadcast_message = write_timeline_service.broadcast_to_followers(post_id, followers)
    print(broadcast_message)

    return post_id

def read_timeline(user):
    """Read Timeline Workflow with Cache Check"""
    memcached = Memcached()
    mongo_db = MongoDB()
    
    post_id = memcached.read_timeline(user)

    if post_id:
        print(f"✅ Cache Hit: Returning {post_id}")
        return post_id
    else:
        print("❌ Cache Miss! Fetching from MongoDB...")
        return mongo_db.read_post(f"POST_{random.randint(1000, 9999)}")

# ------------------------------
# Example Execution
# ------------------------------
post_id = compose_post(
    "Alice", 
    "Hello, world!", 
    media="image.jpg", 
    tags=["Bob", "Charlie"], 
    followers=["Dave", "Eve"]
)
read_timeline("Alice")
