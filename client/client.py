import sys, json, requests

BACKEND = "http://localhost:8000/chat"

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or input("📝 Prompt> ")

    try:
        resp = requests.post(BACKEND, json={"prompt": prompt})
    except requests.exceptions.RequestException as e:
        print(f"Failed to contact backend: {e}")
        sys.exit(1)

    print(f"Status code: {resp.status_code}")
    print(f"Response text: {resp.text}")

    if resp.status_code == 200:
        try:
            data = resp.json()
        except ValueError:
            print("Backend returned invalid JSON")
            sys.exit(1)

        print("\n💡 Top‑5 retrieved prompts:")
        for i, t in enumerate(data.get("similar", []), 1):
            print(f"{i}. {t[:120]}…")
        print("\n📝 Original answer:\n" + data.get("original_answer", ""))
        print("\n🚀 Augmented answer:\n" + data.get("final_answer", ""))
    else:
        print(f"Error: {resp.text}")
