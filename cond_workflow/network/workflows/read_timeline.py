# workflows/read_timeline.py
from services.mongodb import MongoDB
from services.memcached import Memcached

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
