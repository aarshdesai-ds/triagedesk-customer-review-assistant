from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

DB_PATH = "triagedesk.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)

checkpointer = SqliteSaver(conn = conn)

def config_for(review_id):
    review = f"review-{review_id}"
    return {"configurable": {'thread_id': review}}