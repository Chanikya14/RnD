# workflows/post_workflow.py
import random
from services.mongodb import MongoDB
from services.memcached import Memcached
from services.media_services import MediaService
from services.user_tag_service import UserTagService
from services.write_timeline_service import WriteTimelineService
from services.post_storage_service import PostStorageService

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
