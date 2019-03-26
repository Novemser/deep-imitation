import grpc
import time
from concurrent import futures
import data_pb2, data_pb2_grpc
import numpy as np
from scripts.poseEstimate import poseEstimate
from scripts.transfer import model_inference
import cv2
import queue

main_q = queue.Queue()
processed_q = queue.Queue()

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = '127.0.0.1'
_PORT = '9090'
from timeit import default_timer as timer

class FormatData(data_pb2_grpc.TransferImageServicer):
    def DoTransfer(self, request, context):
        # get the image from client request
        #img_shape = tuple(request.shape)
        start = timer()
        re_array = np.frombuffer(request.image, np.uint8)
        print('received client request of image array length: ', len(re_array))
        # Convert back the data to original image shape.
        re_img = cv2.imdecode(re_array, 1)
        #while main_q.qsize() > 10:
        #     main_q.get()
        main_q.put(re_img)
        encimg = processed_q.get()
        #if main_q.empty():
        #     main_q.put(re_img)
        #if not processed_q.empty():
        #     encimg = processed_q.get()
        #else:
        #     encimg = cv2.resize(re_img, (512, 512))
        #     result, encimg = cv2.imencode('.jpg', encimg)
        #print('doing pose estimation...')
        print('Returning to client...')
        res = data_pb2.Data(image=encimg.tobytes())
        end = timer()
        print('used time:', end - start, 'res len:', len(res.image))
        return res

#blank_img = np.zeros((480, 480, 3), np.uint8)
blank_img = cv2.imread('/data1/deeplearning/deep-imitation/server-backend/data/girl1_1_danceinit.png')
blank_img = blank_img[:, :, ::-1]
def serve():
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    data_pb2_grpc.add_TransferImageServicer_to_server(FormatData(), grpcServer)
    grpcServer.add_insecure_port(_HOST + ':' + _PORT)
    grpcServer.start()
    print('GRPC server started at:', _PORT)
    try:
        while True:
            re_img = main_q.get()
            print('doing pose estimation of image shape ', re_img.shape)
            try:
                pose = poseEstimate(re_img)
                print('inferencing...')
                #if pose == -1:
                #    img = blank_img
                #else:
                img = model_inference([pose])[0]
                result, encimg = cv2.imencode('.jpg', img)
                print('Generated image of shape:', img.shape)
                processed_q.put(encimg)
            except IndexError as e:
                print('Detected index err.', e)
                result, re_img = cv2.imencode('.jpg', blank_img)
                processed_q.put(re_img)
#            except ValueError as e:
#                print('Detected index err.', e)
#                result, re_img = cv2.imencode('.jpg', blank_img)
#                processed_q.put(re_img)
            #time.sleep(_ONE_DAY_IN_SECONDS)
            print('enter next round...')
    except KeyboardInterrupt:
        grpcServer.stop(0)

if __name__ == '__main__':
    serve()
