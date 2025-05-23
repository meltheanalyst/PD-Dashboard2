import pandas as pd
import random
from datetime import datetime, timedelta

# Define constants
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 1, 31, 23, 59, 59)
seconds_in_month = int((end_date - start_date).total_seconds())
num_logs = 500000

# Define possible values
products = ["AI Virtual Assistant", "Prototyping Solutions", "Software Assistance"]
product_weights = [0.40, 0.35, 0.25]  # Distribution of inquiries
urls = ["/virtualassistant.php", "/prototype.php", "/softwareassistance.php", "/scheduledemo.php", "/event.php"]
url_to_product = {
    "/virtualassistant.php": "AI Virtual Assistant",
    "/prototype.php": "Prototyping Solutions",
    "/softwareassistance.php": "Software Assistance",
    "/scheduledemo.php": random.choice(products),  # Demo can apply to any product
    "/event.php": random.choice(products)         # Event can apply to any product
}
request_types = ["Inquiry", "Job Request", "Demo Request", "Event Sign-Up"]
request_weights = [0.70, 0.10, 0.15, 0.05]  # Realistic sales funnel
countries = ["UK", "USA", "Germany", "India", "China", "Japan", "Canada", "Australia"]
country_weights = [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.03, 0.02]
response_codes = [200, 304, 404]
response_weights = [0.80, 0.15, 0.05]
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/96.0.4664.45",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0) Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) Firefox/89.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/91.0.864.48"
]
deal_values = {
    "Prototyping Solutions": 5000.00,
    "AI Virtual Assistant": 3000.00,
    "Software Assistance": 2000.00
}

# Generate dataset
data = {
    "Timestamp": [start_date + timedelta(seconds=random.randint(0, seconds_in_month)) for _ in range(num_logs)],
    "IP_Address": [f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}" for _ in range(num_logs)],
    "Request_Method": ["GET"] * num_logs,  # All requests are GET, per IIS log format
    "Request_URL": [random.choice(urls) for _ in range(num_logs)],
    "Response_Code": [random.choices(response_codes, weights=response_weights, k=1)[0] for _ in range(num_logs)],
    "Product_Type": [url_to_product[url] if url in url_to_product else random.choice(products) for url in [random.choice(urls) for _ in range(num_logs)]],
    "Request_Type": [random.choices(request_types, weights=request_weights, k=1)[0] for _ in range(num_logs)],
    "Country": [random.choices(countries, weights=country_weights, k=1)[0] for _ in range(num_logs)],
    "Estimated_Revenue": [0.0] * num_logs,  # Initialize to 0, updated for job requests
    "User_Agent": [random.choice(user_agents) for _ in range(num_logs)]
}

# Update Product_Type for demo/event URLs and set Estimated_Revenue for job requests
for i in range(num_logs):
    if data["Request_URL"][i] in ["/scheduledemo.php", "/event.php"]:
        data["Product_Type"][i] = random.choices(products, weights=product_weights, k=1)[0]
    if data["Request_Type"][i] == "Job Request":
        data["Estimated_Revenue"][i] = deal_values[data["Product_Type"][i]]

# Create DataFrame and sort by Timestamp
df = pd.DataFrame(data)
df["Timestamp"] = df["Timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
df = df.sort_values("Timestamp")

# Save to CSV
df.to_csv("web_server_logs.csv", index=False)

# Print sample for verification
print(df.head(5).to_string(index=False))