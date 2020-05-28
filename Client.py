# Import socket module
import socket
import json


class Client:
    def __init__(self, ip, port):
        self.s = socket.socket()
        self.port = port
        self.ip = ip
        self.s.connect((self.ip, self.port))

    def get_data(self, que):
        data = self.s.recv(1024).decode('utf-8')
        data = json.loads(data)
        print("In Function: {}".format(data))
        que.put(data)

    def send_data(self, data):
        self.s.send(data.encode('utf-8'))

    def shutdown(self):
        self.s.close()

