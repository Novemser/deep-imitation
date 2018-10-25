import grpc
import data_pb2, data_pb2_grpc
import numpy as np

_HOST = 'localhost'
_PORT = '8080'

def run():
    conn = grpc.insecure_channel(_HOST + ':' + _PORT)
    img = np.ones((2, 2, 3), dtype=np.uint8) * 22
    
    client = data_pb2_grpc.TransferImageStub(channel=conn)
    response = client.DoTransfer(data_pb2.Data(shape=bytes(img.shape)))
    print("received: ", tuple(response.shape))

if __name__ == '__main__':
    run()
