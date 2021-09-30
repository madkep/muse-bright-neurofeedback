from time import sleep
from pynput import keyboard
from pynput.keyboard import Key, Controller

keyboard = Controller()

sleep(5)

keyboard.press(" ")
#keyboard.release(" ")
