# need to rename file

from pymongo import MongoClient
from .abstract_semantic_layer import BaseSemantic
from datetime import datetime
import uuid

import os
import dotenv

dotenv.load_dotenv()
import time

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

    

class MessageSaver(BaseSemantic):
    def __init__(self, db_name = 'text2sql', chat_collection = 'chat_log', sql_collection = 'sql_log'):
        
        user_name = os.getenv('MONGO_DB_USER')
        password = os.getenv('MONGO_DB_PASSWORD')
        host = os.getenv('MONGO_DB_HOST') # Share same host with the app
        port = os.getenv('MONGO_DB_PORT')
        
        url = f"mongodb://{user_name}:{password}@{host}:{port}"
        
        self.client = MongoClient(url)
        self.ensure_database_and_collections(db_name, chat_collection, sql_collection)
        
        
        self.db = self.client[db_name]
        self.chat_collection = self.db[chat_collection]
        self.sql_collection = self.db[sql_collection]
        
    def ensure_database_and_collections(self, db_name: str, *collections):
        try:
            # Access the database
            db = self.client[db_name]
            existing_collections = db.list_collection_names()

            # Check and create collections
            for collection in collections:
                if collection not in existing_collections:
                    db.create_collection(collection)
                    logging.info(f"Collection '{collection}' created in database '{db_name}'.")
                else:
                    logging.info(f"Collection '{collection}' already exists in database '{db_name}'.")
        except Exception as e:
            raise Exception(f"Error in creating database and collections: {e}")
        
    def switch_collection(self, collection_name: str):
        self.collection = self.db[collection_name]    
        
    def add_solver_output(self, output: dict):

        # Check if the solver id exists

        solver_id = output['solver_id']

        if not self.sql_collection.find_one({"_id": solver_id}):
            date_created = datetime.now()
            solver_log = {
                "_id": solver_id,
                "solver_output": [output],
                "date_created": date_created
            }
            self.sql_collection.insert_one(solver_log)
        
        else:
            self.sql_collection.update_one(
                {"_id": solver_id},
                {
                    "$push": {
                        "solver_output": output
                    }
                }
            )


    def create_conversation(self, user_id: str):
        """Create a new conversation with OpenAI-style messages."""
        conversation_id = str(uuid.uuid4())
        date_created = datetime.now()
        conversation = {
            "_id": conversation_id,  # MongoDB primary key
            "user_id": user_id,
            "date_created": date_created,
            "date_updated": date_created,
            "messages": [],  # Start with an empty list
            "solver_ids": []
        }
        self.chat_collection.insert_one(conversation)
        logging.info(f"Conversation created with ID: {conversation_id}")
        return conversation_id
    
    # Need new implementation. A message can have multiple sqls_messages
    def add_message(self, conversation_id: str, messages: list[dict], solver_ids: list[str]):
        """Add a message to a conversation."""
        date_updated = datetime.now()

        # Implement push later
        self.chat_collection.update_one(
            {"_id": conversation_id},
            {
                "$set": {
                    "messages": messages,
                    "solver_ids": solver_ids,
                    "date_updated": date_updated
                }
            }
        )
        logging.info(f"Message added to conversation with ID: {conversation_id}")

    def get_messages(self, conversation_id: str):
        return self.chat_collection.find_one({"_id": conversation_id})
    

    def sql_feedback(self, solver_id: int, output_id: int, feedback):

        self.sql_collection.update_one(
            {"_id": solver_id, "solver_output.output_id": output_id},  # Match your document
            {"$set": {"solver_output.$.feedback": feedback}},  # Update the last message
        )
        # self.sql_collection.update_one(
        #     {"_id": sql_id},  # Match your document
        #     {"$set": {"feedback": feedback}},  # Update the last message
        # )
            
    
def get_semantic_layer(**kwargs):
    try:
        message_saver = MessageSaver(**kwargs)
        return message_saver
    except Exception as e:
        logging.error(f"Error in getting semantic layer: {e}")
        return BaseSemantic()
    
    
if __name__ == "__main__":
    
    message = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
]
    
    user_id = "test_func"
    
    message_saver = MessageSaver()
    conversation_id = message_saver.create_conversation(user_id)
    print(f"Conversation created with ID: {conversation_id}")
    time.sleep(1)
    
    message_saver.add_message(conversation_id, message, [])