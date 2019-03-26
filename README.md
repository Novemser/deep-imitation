##Deep imitation
A real time human pose imitation project based on Everybody Dance Now(https://arxiv.org/abs/1808.07371).

This is currently a work in progress.

###System requirements
- pytorch 0.4.1
- grpc
- python 3.6
- opencv2
- aiortc

You might need to first train your model according to Everybody Dance Now.

###Start backend server
```bash
cd server-backend
python server.py
```

###Start video server
```bash
cd server-video
python server.py --port 8080
```
