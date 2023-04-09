# -*- coding: UTF-8 -*-
#!/usr/bin/python3

import binascii
import serial
import serial.tools.list_ports
import time
import cv2 as cv
from PIL import Image

def recv(serial):
    while True:
        data = serial.read_all()
        if data == '':
            continue
        else:
            break
    return data

def read_fingerprint(serial):
    if serial.isOpen() :
        print("open success")
    else :
        print("open failed")
    while(True):
        a = 'EF 01 FF FF FF FF 01 00 03 01 00 05'
        d = bytes.fromhex(a)
        serial.write(d)
        time.sleep(1)
        data =recv(serial)
        if data != b'' :
            data_con = str(binascii.b2a_hex(data))[20:22]
            if(data_con == '02'):
                print("请按下手指")
            elif(data_con == '00'):
                print("载入成功")
                
                upload = 'EF 01 FF FF FF FF 01 00 03 0a 00 0e'
                uploadH = bytes.fromhex(upload)
                serial.write(uploadH)
                # 读取指令返回值
                data = serial.read(12)
                if data[9:11] == b'\x00\x00':
                # 如果返回值正确，则开始读取图像数据
                    data = serial.read(529)
                    print('读取成功')
                    serial.close()
                    break
    data =  str(binascii.b2a_hex(data))
    print(data)               
    fingerprint = Image.open(data)
    size = fingerprint.shape
    # print(size[0],size[1])
    cv.imshow('Fingerprint', fingerprint)
    return fingerprint








