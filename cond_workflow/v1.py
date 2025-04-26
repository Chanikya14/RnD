import torch
import random
import time

# ------------------------------
# Stage 1: Search & Availability Check
# ------------------------------
def check_availability(hotels):
    """Check if rooms are available."""
    available_hotels = [hotel for hotel in hotels if hotel["rooms"] > 0]
    if available_hotels:
        return available_hotels
    return None

# ------------------------------
# Stage 2: Alternative Hotel Recommendations (GPU)
# ------------------------------
def recommend_alternative_hotels(user_embedding, hotel_embeddings):
    """Use GPU-accelerated cosine similarity for recommendations."""
    user_tensor = torch.tensor(user_embedding, dtype=torch.float32, device="cuda")
    hotel_tensor = torch.tensor(hotel_embeddings, dtype=torch.float32, device="cuda")

    # Compute cosine similarity
    similarity_scores = torch.nn.functional.cosine_similarity(hotel_tensor, user_tensor.unsqueeze(0))
    recommended_indices = torch.argsort(similarity_scores, descending=True)[:5]  # Top 5 recommendations
    
    return recommended_indices.cpu().numpy()

# ------------------------------
# Stage 3: Room Selection & Pricing (GPU)
# ------------------------------
def calculate_discount(prices, discount_rates):
    """Apply discount calculations on GPU."""
    prices = torch.tensor(prices, dtype=torch.float32, device="cuda")
    discount_rates = torch.tensor(discount_rates, dtype=torch.float32, device="cuda")
    
    discounted_prices = prices * (1 - discount_rates)  
    return discounted_prices.cpu().numpy()

# ------------------------------
# Stage 4: Payment Processing (CPU)
# ------------------------------
def process_payment(amount):
    """Simulate a payment process."""
    success = random.choice([True, False])  # Random success/fail
    return success

# ------------------------------
# Stage 5: Confirmation & Notification
# ------------------------------
def send_confirmation(user, hotel, price):
    """Send booking confirmation."""
    print(f"✅ Booking Confirmed for {user} at {hotel} for ${price}")

# ------------------------------
# Main Execution Workflow
# ------------------------------
if __name__ == "__main__":
    # Sample hotels (name, rooms available)
    hotels = [
        {"name": "Hotel A", "rooms": 0}, 
        {"name": "Hotel B", "rooms": 3},
        {"name": "Hotel C", "rooms": 0}
    ]
    
    # Simulated user and hotel embeddings for recommendations
    user_embedding = [0.5, 0.8, 0.3]
    hotel_embeddings = [[0.4, 0.7, 0.2], [0.6, 0.9, 0.4], [0.2, 0.5, 0.1]]

    # Stage 1: Availability Check
    available_hotels = check_availability(hotels)
    print(available_hotels)
    
    if available_hotels:
        print("✅ Rooms available. Proceeding with booking.")
        chosen_hotel = available_hotels[0]["name"]
    else:
        print("❌ No rooms available. Finding alternatives...")
        recommended_hotels = recommend_alternative_hotels(user_embedding, hotel_embeddings)
        chosen_hotel = f"Alternative Hotel {recommended_hotels[0]}"
    
    # Stage 3: Pricing & Discounts
    base_prices = [100, 200, 300]  # Room prices
    discount_rates = [0.1, 0.2, 0.15]  # Discount rates
    final_prices = calculate_discount(base_prices, discount_rates)
    final_price = final_prices[0]
    
    # Stage 4: Payment Processing
    print(f"💳 Processing payment of ${final_price}...")
    time.sleep(1)  # Simulate delay
    if process_payment(final_price):
        # Stage 5: Confirmation
        send_confirmation("User123", chosen_hotel, final_price)
    else:
        print("❌ Payment failed! Please try again.")
