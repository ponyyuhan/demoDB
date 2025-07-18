import sys, json, requests

BACKEND = "http://localhost:8000/chat"

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or input("📝 Prompt> ")
    resp = requests.post(BACKEND, json={"prompt": prompt})
    data = resp.json()
    print("\n💡 Top‑5 retrieved prompts:")
    for i, t in enumerate(data["similar"], 1):
        print(f"{i}. {t[:120]}…")
    print("\n🤖 Answer:\n" + data["answer"])
