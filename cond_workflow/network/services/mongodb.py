# services/mongodb.py
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
