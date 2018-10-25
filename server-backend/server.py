import grpc
import time
from concurrent import futures
import data_pb2, data_pb2_grpc
import numpy as np

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = 'localhost'
_PORT = '8080'

class FormatData(data_pb2_grpc.TransferImageServicer):
    def DoTransfer(self, request, context):
        sp = tuple(request.shape)
        print('received client request of image:shape ', sp)
        img = np.ones((50, 50, 3), dtype=np.uint8) * 222
        img_shape = bytes(img.shape)

        return data_pb2.Data(shape=img_shape, image=np.ndarray.tobytes(img))

def serve():
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    data_pb2_grpc.add_TransferImageServicer_to_server(FormatData(), grpcServer)
    grpcServer.add_insecure_port(_HOST + ':' + _PORT)
    grpcServer.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        grpcServer.stop(0)

if __name__ == '__main__':
    serve()
