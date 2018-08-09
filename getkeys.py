#Get Keys
# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

import win32api as wapi
import win32con as w
import time

keyList = ["\b"]
specialKeys = {w.VK_DOWN: "DOWN", w.VK_RETURN: "ENTER"}
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

def key_check():
    
    keys = []
    
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    for keyCode, keyName in specialKeys.items():
        if wapi.GetAsyncKeyState(keyCode):
           keys.append(keyName)
           
    return keys

 
