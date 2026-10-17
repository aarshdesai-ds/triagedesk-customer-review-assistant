from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

DB_PATH = "triagedesk.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)

checkpointer = SqliteSaver(conn = conn)

def config_for(review_id):
    review = f"review-{review_id}"
    return {"configurable": {'thread_id': review}}

if __name__ == "__main__":
    import os

    config = config_for("L01")
    print("config      ", config)
    print("ok" if config == {"configurable": {"thread_id": "review-L01"}} else "FAIL", "config shape")
    print("ok" if os.path.exists(DB_PATH) else "FAIL", f"{DB_PATH} exists")

    saved = list(checkpointer.list(config))
    print(f"checkpoints for review-L01: {len(saved)}  (0 is expected before any run)")