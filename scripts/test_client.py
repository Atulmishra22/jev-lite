import httpx
import json

BASE_URL = "http://localhost:8000/"

payload = {
    "model": "jev-lite-0.1",
    "state": {
        "customer": "Alice",
        "ticket": "My credit card was charged twice ($49.99 x 2) for order #A-1014. Please refund the duplicate immediately.",
        "order_status": "delivered",
        "history": "loyal customer for 3 years, 0 prior disputes."
    },
    "questions":{
        "department" : {
            "type" : "choice",
            "instructions" : "which department should handle this ticket?",
            "criteria" : {
                "billing" : "Issues with payments, credit card charges or refunds",
                "tech_support" : "issues with login, app crashes or software bugs",
                "shipping" : "Issues with package delivery, tracking or postal damage"
            }

        },
        "urgency": {
            "type": "score",
            "instructions": "rate the customer urgency level",
            "levels" :{
                "1": "Low - General questions or feedback",
                "2": "Medium - Minor bug or inquiry",
                "3": "High - Financial dispute or blocker"
            }
        },
        "is_refund":{
            "type": "noul",
            "statement": "the customer is explicitly asking for money to be refunded, indicating a refund request."
        }
    }
}

def run_test():
    print(f" 1. checin health ({BASE_URL}v1/health)....")
    res = httpx.get(f"{BASE_URL}v1/health")
    print("Health check response:", res.json())

    print("\n 2. sending system one parallel evaluation request...")
    res = httpx.post(f"{BASE_URL}/v1/systemone", json=payload, timeout=60.0)

    if res.status_code == 200:
        print("\n Response received Sucessfully:")
        print(json.dumps(res.json(), indent=4))
    else:
        print(f"\n Request failed with status code: {res.status_code}")
        print("Response:", res.text)

if __name__ == "__main__":
    run_test()