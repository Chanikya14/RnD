# services/media_service.py
class MediaService:
    """Handles media processing (images/videos)"""
    def process_media(self, post_id, media_type):
        """Process media attached to post"""
        return f"📷 Media Processing for {post_id}: {media_type}"
