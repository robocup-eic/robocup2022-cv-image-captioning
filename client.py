import socket
import cv2
import numpy as np
import time
from custom_socket import CustomSocket
import json

image = cv2.imread("dataset/t11.jpg")

print(image.shape)
image = cv2.resize(image, (1280, 720))

host = socket.gethostname()
# host = "192.168.8.2"
port = 10008

c = CustomSocket(host, port)
c.clientConnect()
print("Send")
msg = c.req(image)
print(msg)

# cv2.imshow(" ", image)
# cv2.waitKey()
