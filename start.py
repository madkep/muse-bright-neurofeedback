from muselsl import stream, list_muses

muses = list_muses('bleak')
stream(muses[0]['address'],backend='bleak',preset=22)

# Note: Streaming is synchronous, so code here will not execute until after the stream has been closed
print('Stream has ended')