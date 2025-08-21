SYSTEM = (
    "You are a data-to-text model. "
    "Given a table and highlighted cells, write ONE faithful sentence."
)
RESPONSE_TAG = "### OUTPUT:\n"   # TRL will treat everything AFTER this as labels

def format_example(ex):
    prompt = f"{SYSTEM}\n### TABLE\n{ex['text']}\n{RESPONSE_TAG}{ex['labels']}"
    return {"text": prompt}
