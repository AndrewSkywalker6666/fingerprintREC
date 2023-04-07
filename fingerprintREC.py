# -*- coding: UTF-8 -*-
#!/usr/bin/python3

import binascii
import serial
import serial.tools.list_ports
import time
import struct

def recv(serial):
    while True:
        data = serial.read_all()
        if data == '':
            continue
        else:
            break
    return data

if __name__ == '__main__':
    # BMP文件类型标识
    bm_magic = 0x4D42
    # BMP文件头
    pixel_data_size = 640 * 480
    file_size = 54 + pixel_data_size  # 文件大小
    reserved1 = 0                    # 保留字段1
    reserved2 = 0                    # 保留字段2
    pixel_data_offset = 54           # 像素数据偏移量

    # BMP信息头（DIB头）
    dib_size = 40                    # DIB头大小
    image_width = 640                # 图像宽度
    image_height = 480               # 图像高度
    num_planes = 1                   # 位面数
    bits_per_pixel = 24              # 每个像素的位数
    compression = 0                  # 压缩类型（0表示未压缩）
    image_size = 0                   # 像素数据大小
    h_resolution = 0                 # 水平分辨率
    v_resolution = 0                 # 垂直分辨率
    num_colors = 0                   # 颜色数
    important_colors = 0             # 重要颜色数

    # 构建BMP文件头
    bmp_header = struct.pack('<HL2HL', bm_magic, file_size, reserved1, reserved2, pixel_data_offset)
    bmp_header += struct.pack('<L3l2L2l', dib_size, image_width, image_height, num_planes, bits_per_pixel, compression, image_size, h_resolution)

    serial = serial.Serial('/dev/ttyUSB0', 57600, timeout=0.5)  #/dev/ttyUSB0
    if serial.isOpen() :
        print("open success")
    else :
        print("open failed")
    while True:
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
                    print('read success')
                # 将图像数据保存到文件中
                    with open('fingerprint.img', 'wb') as f:
                        f.write(data)
                    exit()
                buff = 'EF 01 FF FF FF FF 01 00 04 02 01 00 08'
                buff = bytes.fromhex(buff)
                serial.write(buff)
                time.sleep(1)
                buff_data = recv(serial)
                buff_con = str(binascii.b2a_hex(buff_data))[20:22]
                if(buff_con == '00'):
                    print("生成特征成功")
                    serch = 'EF 01 FF FF FF FF 01 00 08 04 01 00 00 00 64 00 72'
                    serch = bytes.fromhex(serch)
                    serial.write(serch)
                    time.sleep(1)
                    serch_data = recv(serial)
                    serch_con = str(binascii.b2a_hex(serch_data))[20:22]
                    if (serch_con == '09'):
                        print("指纹不匹配")
                    elif(serch_con == '00'):
                        print("指纹匹配成功")
                serial.close()
                #exit()
            else:
                print("不成功")

