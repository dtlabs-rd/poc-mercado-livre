import pika
import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()


@dataclass
class Config:
    username: str
    password: str
    host: str
    credentials: pika.PlainCredentials

    def __init__(self):
        self.username = os.getenv("RABBITMQ_USER")
        self.password = os.getenv("RABBITMQ_PASS")
        self.host = os.getenv("RABBITMQ_HOST")
        self.credentials = pika.PlainCredentials(
            username=self.username, password=self.password)

    def get_credentials(self) -> pika.PlainCredentials:
        return self.credentials


class ConnectionManager:
    _connection = None

    @staticmethod
    def get_connection(config: Config):
        if ConnectionManager._connection is None or ConnectionManager._connection.is_closed:
            print("Creating connection")
            ConnectionManager._connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=config.host,
                    credentials=config.get_credentials(),
                    heartbeat=300,
                    blocked_connection_timeout=300,
                    client_properties={'connection_name' : 'POC-MELI-DEVICE'}
                )
            )
        return ConnectionManager._connection

    @staticmethod
    def create_channel(config: Config):
        connection = ConnectionManager.get_connection(config)
        return connection.channel()

    @staticmethod
    def close_connection():
        if ConnectionManager._connection is not None and not ConnectionManager._connection.is_closed:
            ConnectionManager._connection.close()

class RabbitMQ:
    def __init__(self, config: Config, exchange: str, exchange_type: str):
        self.config = config
        self.exchange = exchange
        self.exchange_type = exchange_type
        self._connection_manager = ConnectionManager()
        self._channel = self.create_channel()
        
    def get_connection(self):
        return self._connection_manager.get_connection(self.config)

    def close_connection(self):
        self._connection_manager.close_connection()
        
    def create_channel(self):
        channel = self.get_connection().channel()
        channel.exchange_declare(self.exchange, self.exchange_type)
        return channel

    def get_channel(self):   
        return self._channel
        

