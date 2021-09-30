from muselsl import stream, list_muses
import time

start = time.time()
while(True):
    muses = list_muses()
    stream(muses[0]['address'],preset=22)
    # Note: Streaming is synchronous, so code here will not execute until after the stream has been closed
    print(start)
print('Stream has ended')