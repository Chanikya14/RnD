# services/post_storage_service.py
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
