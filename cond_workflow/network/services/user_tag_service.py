# services/user_tag_service.py
class UserTagService:
    """Handles user tagging in posts"""
    def handle_tags(self, post_id, tags):
        """Process user tags in a post"""
        return f"🏷️ Tags Added for {post_id}: {', '.join(tags)}"
