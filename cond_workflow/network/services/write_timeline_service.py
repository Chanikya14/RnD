# services/write_timeline_service.py
class WriteTimelineService:
    """Broadcasts post to user’s followers"""
    def broadcast_to_followers(self, post_id, followers):
        """Broadcast the post to all followers"""
        return f"📣 Broadcasted post {post_id} to followers: {', '.join(followers)}"
