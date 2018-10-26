import argparse
import asyncio
import json
import logging
import os

import cv2
from aiohttp import web
from av import VideoFrame

from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

import grpc
import data_pb2, data_pb2_grpc
import numpy as np
import pickle
from grpc._cython.cygrpc import CompressionAlgorithm
from grpc._cython.cygrpc import CompressionLevel
chan_ops = [('grpc.default_compression_algorithm', CompressionAlgorithm.gzip),
            ('grpc.grpc.default_compression_level', CompressionLevel.high)]
_HOST = '127.0.0.1'
_PORT = '10000'
conn = grpc.insecure_channel(_HOST + ':' + _PORT, options=chan_ops)
client = data_pb2_grpc.TransferImageStub(channel=conn)

ROOT = os.path.dirname(__file__)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]

def transfer(img):
    #img = np.ones((2, 2, 3), dtype=np.uint8) * 22
    print(1)
    # img_bytes = np.ndarray.tobytes(img)
    img_bytes = img.tobytes()
    print(2)
    #client = data_pb2_grpc.TransferImageStub(channel=conn)
    # np.array(img.shape)
    data = data_pb2.Data(image=img_bytes)
    # print('start transfer img via gRPC of shape:', img.shape)
    print(3)
    response = client.DoTransfer(data)
    print(4)
    # img_shape = tuple(response.shape)
    img_new = cv2.imdecode(np.fromstring(response.image, np.uint8), 1)
    print("received transfered image: ", tuple(img_new.shape))
    # re_img = np.reshape(img_new, img_shape)
    return img_new

class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.counter = 0
        self.track = track
        self.transform = transform

    async def recv(self):
        frame = await self.track.recv()
        self.counter += 1

        if self.transform == 'edges':
            # perform edge detection
            img = frame.to_ndarray(format='bgr24')
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format='bgr24')
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == 'rotate':
            # rotate image
            img = frame.to_ndarray(format='bgr24')
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format='bgr24')
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        else:
            img = frame.to_ndarray(format='bgr24')
            result, encimg = cv2.imencode('.jpg', img, encode_param)
            generated = transfer(encimg)
            
            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(generated, format='rgb24')
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame



async def index(request):
    content = open(os.path.join(ROOT, 'index.html'), 'r').read()
    return web.Response(content_type='text/html', text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, 'client.js'), 'r').read()
    return web.Response(content_type='application/javascript', text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(
        sdp=params['sdp'],
        type=params['type'])

    pc = RTCPeerConnection()
    pcs.add(pc)

    # prepare local media
    player = MediaPlayer(os.path.join(ROOT, 'demo-instruct.wav'))
    if args.write_audio:
        recorder = MediaRecorder(args.write_audio)
    else:
        recorder = MediaBlackhole()

    @pc.on('datachannel')
    def on_datachannel(channel):
        @channel.on('message')
        def on_message(message):
            channel.send('pong')

    @pc.on('iceconnectionstatechange')
    async def on_iceconnectionstatechange():
        print('ICE connection state is %s' % pc.iceConnectionState)
        if pc.iceConnectionState == 'failed':
            await pc.close()
            pcs.discard(pc)

    @pc.on('track')
    def on_track(track):
        print('Track %s received' % track.kind)

        if track.kind == 'audio':
            pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == 'video':
            local_video = VideoTransformTrack(track, transform=params['video_transform'])
            pc.addTrack(local_video)

        @track.on('ended')
        async def on_ended():
            print('Track %s ended' % track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type='application/json',
        text=json.dumps({
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        }))


pcs = set()


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WebRTC audio / video / data-channels demo')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port for HTTP server (default: 8080)')
    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('--write-audio', help='Write received audio to a file')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get('/', index)
    app.router.add_get('/client.js', javascript)
    app.router.add_post('/offer', offer)
    web.run_app(app, port=args.port)
