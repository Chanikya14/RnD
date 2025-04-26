# main.py
from workflows.post_workflow import compose_post
from workflows.read_timeline import read_timeline

# Example of post creation
post_id = compose_post(
    "Alice", 
    "Hello, world!", 
    media="image.jpg", 
    tags=["Bob", "Charlie"], 
    followers=["Dave", "Eve"]
)

# Example of reading the timeline
read_timeline("Alice")
