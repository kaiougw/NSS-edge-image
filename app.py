#若直接執行本程式，程式會將子目錄中的BMP檔案重新命名並搬出資料夾，對每個BMP圖檔數值化解析，並輸出解析圖
#需要NSS的檔案是LOT@slot@datetime@300RXM06@EDL_2，沒有_2的不適用
#若不須重解壓縮，新命名及搬運的功能，則by pass "decompress()"這個function即可

import py7zr
import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean, sqrt, square, arange
import math
import scipy.signal
import glob
import csv
import fnmatch
import pandas as pd
import streamlit as st
from pathlib import Path
import zipfile
import traceback

#功能: 讀取指定的excel，並將欄寬自動最佳化 (Automatically adjust column widths in Excel)
#使用方式: 先指定name_of_wb為想要自動化的workbook檔案名稱
def auto_fit(name_of_wb=''): # Input: Excel file name
    if name_of_wb !='':   
        from openpyxl import load_workbook
        from openpyxl.utils import get_column_letter

        #讀取已經存在的特定檔案
        wb=load_workbook(name_of_wb) #()裡面是完整的檔案名稱
        #對每個分頁做auto_fit
        for i in wb.sheetnames:
            ws=wb[i]
            #auto_fit:
            for letter_num in range(1,ws.max_column+1):
                max_width=0
                letter=get_column_letter(letter_num)#數字轉A, B, C.....AA, AB, AC......
                for row_number in range(1, ws.max_row+1):
                    #用try避免讀到空白時跳error卡住
                    try: #取特定欄位A1的值，要寫ws['A1'].value，依此類推
                        if len(ws[f'{letter}{row_number}'].value)>max_width: # update max_width if the length of cell value is longer than current max_width
                            max_width=len(ws[f'{letter}{row_number}'].value)
                    except: 
                        pass
                ws.column_dimensions[letter].width=(max_width+2)*1.2
        wb.save(name_of_wb)



def moving_average(x, w):
    """
    Input:
        x: 1D array
        w: window size (integer)
    """
    return np.convolve(x, np.ones(w), 'valid') / w

def rolling_window(a, window):
    """
    Input:
        a: 1D array
        window: window size (integer)
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
def find_nonblack(npy1d,threshold):
    """
    Input:
        npy1d: 1D array
        threshold: value to compare against (integer)
    """
    peaks = scipy.signal.find_peaks_cwt(npy1d,10)
    #plt.plot(col)
    #plt.plot(peaks ,col[peaks],'o')
    #plt.show()
    for p in peaks:
        if npy1d[p]>=threshold:
            res=p
            break
    return res


def convert_nss_rawimage(img_file):
    """
    Process a raw NSS BMP image and detect the notch position to reconstruct the full wafer edge image.
    Input:
        img_file: path to the BMP image file (string)
    Output:
        img_v1: processed image as a 2D array
    """
    #img_file='.\\P1276BM\\mjl\\1-9 (SILTRONIC-WF)_EDU_0.bmp'
    img_v = cv2.imread( img_file,cv2.IMREAD_GRAYSCALE)
    #print(img_v.shape)
    
    if img_v.shape[0]==384000 and img_v.shape[1]==512:
        col_u=img_v[:,5] #detection upper notch white spot
        col_l=img_v[:,507] #detection lower notch white spot
        res_u=0
        res_l=0
        
        for x, val in enumerate(col_u): # scan upper column for first pixel >=254
            if val>=254:
                res_u=x
                break
        for x, val in enumerate(col_l): # scan lower column for first pixel >=254
            if val>=254:
                res_l=x
                break       
        
        res=np.max([res_u,res_l]) # tkae upper or lower detected position as notch position
        
        if res>0: # if notch is found, proceed to reconstruct image and slice image into two parts 
            end_1=int((img_v.shape[0]-res)/1038)-2
            top_1=(359-end_1+1)
            img_1=img_v[res:res+end_1*1038,:]
            img_2=img_v[res-top_1*1038:res,:]
            
            img_v1 = cv2.vconcat([img_1, img_2]) # concatenate two parts vertically to form a continuous wafer image
            #print(img_v1.shape)
            for i in range(0,360):
                #cv2.line(img_v1, (0,i*1038), (45,i*1038), (255, 255, 255), 3)
                #cv2.putText(img_v1, str(i), (10,i*1038+30), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 255, 255), 1, cv2.LINE_AA)
                
                cv2.line(img_v1, (img_v1.shape[1]-45,i*1038), (img_v1.shape[1],i*1038), (255, 255, 255), 3)
                cv2.putText(img_v1, str(i), (img_v1.shape[1]-60,i*1038+30), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 255, 255), 1, cv2.LINE_AA)
            
            img_v1=cv2.rotate(img_v1, cv2.ROTATE_90_CLOCKWISE)
            img_v1=cv2.flip(img_v1, 1) #flip left-right
            #cv2.imwrite('.\\P1276BM\\mjl\\test.png', img_v1)
            return img_v1 
        else:
            print("Can't find the notch.")
            return np.array([])
    else:
        print("Invalid Image.")
        return np.array([])

def turning_points(array):
    ''' 
    Finds the turning points within an 1D array and returns the indices of the minimum and 
    maximum turning points in two separate lists.

    Input:
        array: 1D array
    Output:
        idx_min: list of indices of minimum turning points
        idx_max: list of indices of maximum turning points
    '''
    idx_max, idx_min = [], []
    if (len(array) < 3): 
        return idx_min, idx_max

    NEUTRAL, RISING, FALLING = range(3)
    def get_state(a, b):
        if a < b: return RISING
        if a > b: return FALLING
        return NEUTRAL

    ps = get_state(array[0], array[1])
    begin = 1
    for i in range(2, len(array)): # if trend changes (from rising to falling or falling to rising), record turning point
        s = get_state(array[i - 1], array[i]) # compare each pair of values
        if s != NEUTRAL:
            if ps != NEUTRAL and ps != s:
                if s == FALLING: 
                    idx_max.append((begin + i - 1) // 2)
                else:
                    idx_min.append((begin + i - 1) // 2)
            begin = i
            ps = s
    return idx_min, idx_max

def process_bmp0(bmpfile):
    """
    Process a BMP file to analyze wafer edge image, compute roughness, and generate related charts and CSV files.

    Input:
        bmpfile: path to the BMP image file (string)
    Output:
        List containing analysis results
    """
    #print(bmpfile)

    nss_img_path= os.path.dirname(bmpfile) +'/'
    img_file=bmpfile#'.//P1276BM//' + f + '.png'
    img_v=convert_nss_rawimage(img_file)

    #print(nss_img_path + os.path.basename(bmpfile)[:-4])

    if img_v.shape[1]<(360*1038):
        print('Invalid NSS bmp file...!')
        return []
    else:
        #detection wafer position 
        col1=img_v[:,5000]
        col2=img_v[:,5500]
        col3=img_v[:,10000]
        col4=img_v[:,10500]
        col5=img_v[:,20000]
        col6=img_v[:,21000]
        th_=50
        res1 = find_nonblack(col1,th_)
        res2 = find_nonblack(col2,th_)
        res3 = find_nonblack(col3,th_)
        res4 = find_nonblack(col4,th_)
        res5 = find_nonblack(col5,th_)
        res6 = find_nonblack(col6,th_)
        res=np.min([res1,res2,res3,res4,res5,res6])
        c1=res+10
        c2=res+230
        img_v0=img_v[c1:c2,10000:10500]
        #plt.imshow(img_v0,cmap=plt.cm.gray)
        #plt.show()
        #plt.clf()
        
        #column stdev
        a0=np.std(img_v[c1:c2,:],axis=0)
        x= np.arange(0,a0.shape[0])
        
        #row stdev 1038
        mv_sdev_window=1038 #519
        a01=np.std(rolling_window(a0, mv_sdev_window), 1)**2  #1038
        ax=moving_average(x/1038, mv_sdev_window) #1038

        #Ra & Q95
        a02=a01[1038*3:1038*-2]
        Ra=np.sum(np.abs(a02))/a02.shape[0]
        Q95=np.percentile(a02, 95, axis=0)
        Q90=np.percentile(a02, 90, axis=0)
        Q50=np.percentile(a02, 50, axis=0)
        #rpt.append([img_file,Ra,Q50,Q90,Q95])

        
        #360 ra
        rpt_360=[]
        for i in range(3,358):
            a_=a01[1038*i:1038*(i+1)]
            ra_=np.sum(np.abs(a_))/a_.shape[0]
            rpt_360.append([i,ra_])
        
        #find peaks
        a_360_x=np.array(rpt_360)[:,0]
        a_360_y=np.array(rpt_360)[:,1]
        idx_min,idx_max=turning_points(a_360_y)
        rpt_360_peaks=[]
        for i in idx_max:
            rpt_360_peaks.append([a_360_x[i],a_360_y[i]])
        arr=np.array(rpt_360_peaks)
        a_360_peaks=arr[arr[:, 1].argsort()]

        
        #plot by deg chart
        plt.figure(figsize=(8,4))
        plt.title('NSS EDGE Image Quality Ra by Deg')
        plt.plot(a_360_x,a_360_y,label=img_file)
        plt.scatter(a_360_peaks[-10:,0],a_360_peaks[-10:,1],facecolors='none', edgecolors='r')
        plt.scatter(a_360_peaks[-20:-10,0],a_360_peaks[-20:-10,1],facecolors='none', edgecolors='g')
        plt.ylim(0,100)
        plt.xticks(np.arange(0, 360, 10.0))
        plt.xticks(rotation=90, ha='left')
        plt.xlabel('Angle NotchDown CW')
        plt.legend()
        plt.savefig(nss_img_path + os.path.basename(bmpfile)[:-4] + '_360chart.jpg')
        #plt.show()
        plt.clf()
        plt.close()

        #excluded notch before 2500 and last 2500 pixels
        plt.figure(figsize=(8,4))
        plt.title('NSS EDGE Image Quality Ra=' + str(round(Ra,2)))
        plt.plot(ax[1038*3:1038*-2],a01[1038*3:1038*-2],label=img_file)
        plt.ylim(0,100)
        plt.xlabel('Angle NotchDown CW')
        plt.xticks(np.arange(0, 360, 10.0))
        plt.xticks(rotation=90, ha='left')
        plt.legend()
        plt.savefig(nss_img_path + os.path.basename(bmpfile)[:-4] + '_chart.jpg')
        #plt.show()
        plt.clf()
        plt.close()
        #export to save csv files
        np.savetxt(nss_img_path + os.path.basename(bmpfile)[:-4] + '_peaks.csv',a_360_peaks, fmt='%1.3f',delimiter=',')
        np.savetxt(nss_img_path + os.path.basename(bmpfile)[:-4] + '_360.csv',np.array(rpt_360), fmt='%1.3f',delimiter=',')
        #np.savetxt(nss_img_path + os.path.basename(f)[:-4] + '.csv',a01, fmt='%1.3f',delimiter=',')


        #save images
        for i in np.arange(1038*3,a0.shape[0]-1038*2):
            x0=int(i)+519
            x1=int(i+1)+519
            y0=int(a01[i]/100*200)
            y1=int(a01[i+1]/100*200)
            cv2.line(img_v, (x0, img_v.shape[0]-y0-10), (x1, img_v.shape[0]-y1-10), (255, 255, 255), 1)

        #print(nss_img_path + os.path.basename(f)[:-4] + '.png')
        #save whole wafer image
        cv2.imwrite(nss_img_path + os.path.basename(bmpfile)[:-4] + '.png', img_v)

        #crop image for top 20 peaks
        tmp_dir=os.path.dirname(bmpfile) + '/' + os.path.basename(bmpfile)[:-4]
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        for i in a_360_peaks[-20:,0]:
            img_v0=img_v[:,int((i-0.1)*1038):int((i+2)*1038)]
            cv2.imwrite(tmp_dir + '/' + str(int(i)) + '.png', img_v0)        

        return [img_file,Ra,Q50,Q90,Q95]

def process_bmp(bmpfile):
    #print(bmpfile)
    f=bmpfile
    nss_img_path= os.path.dirname(bmpfile) +'/'
    img_file=bmpfile#'.//P1276BM//' + f + '.png'
    img_v=convert_nss_rawimage(img_file)

    #print(nss_img_path + os.path.basename(bmpfile)[:-4])
    #row stdev 1038
    #mv_sdev_window_0=5 #519 short wave
    mv_sdev_window_1=1038 #519 long wave
        
    if img_v.shape[1]<(360*mv_sdev_window_1):
        print('Invalid NSS bmp file...!')
        return []
    else:
        #detection wafer position 
        col1=img_v[:,5000]
        col2=img_v[:,5500]
        col3=img_v[:,10000]
        col4=img_v[:,10500]
        col5=img_v[:,20000]
        col6=img_v[:,21000]
        th_=50
        res1 = find_nonblack(col1,th_)
        res2 = find_nonblack(col2,th_)
        res3 = find_nonblack(col3,th_)
        res4 = find_nonblack(col4,th_)
        res5 = find_nonblack(col5,th_)
        res6 = find_nonblack(col6,th_)
        res=np.min([res1,res2,res3,res4,res5,res6])
        c1=res+10
        c2=res+230
        #img_v0=img_v[c1:c2,10000:10500]
        #plt.imshow(img_v0,cmap=plt.cm.gray)
        #plt.show()
        #plt.clf()
        
        #column stdev
        a0_0=np.max(img_v[c1:c2,:],axis=0)-np.min(img_v[c1:c2,:],axis=0)
        a0_1=np.std(img_v[c1:c2,:],axis=0)
        x= np.arange(0,a0_1.shape[0])
        
        
        #a01_0=a0_0-np.mean(a0_0)  #1038
        a01_0=a0_0-np.min(a0_0[mv_sdev_window_1*3:mv_sdev_window_1*-3])  #1038
        a01_1=np.std(rolling_window(a0_1, mv_sdev_window_1), 1)**2  #1038
        #angles
        ax_0=x/mv_sdev_window_1 #1038
        ax_1=moving_average(x/mv_sdev_window_1, mv_sdev_window_1) #1038
        
        #Ra & Q95
        a02_0=a01_0[mv_sdev_window_1*3:mv_sdev_window_1*-3]
        a02_1=a01_1[mv_sdev_window_1*3:mv_sdev_window_1*-3]


        Ra_0=np.sum(np.abs(a02_0))/a02_0.shape[0]
        Q99_0=np.percentile(a02_0, 99, axis=0)
        Q90_0=np.percentile(a02_0, 90, axis=0)
        Q50_0=np.percentile(a02_0, 50, axis=0)
        
        Ra_1=np.sum(np.abs(a02_1))/a02_1.shape[0]
        Q99_1=np.percentile(a02_1, 99, axis=0)
        Q90_1=np.percentile(a02_1, 90, axis=0)
        Q50_1=np.percentile(a02_1, 50, axis=0)

        
        #rpt.append([img_file,Ra_0,Q50_0,Q90_0,Q99_0,Ra_1,Q50_1,Q90_1,Q99_1])

        
        #360 ra
        rpt_360=[]
        for i in range(3,358):
            a_0=a01_0[mv_sdev_window_1*i:mv_sdev_window_1*(i+1)]
            a_1=a01_1[mv_sdev_window_1*i:mv_sdev_window_1*(i+1)]

            a_0_98=np.percentile(a_0, 98, axis=0)
            ra_0=np.mean(a_0_98)
            ra_1=np.sum(np.abs(a_1))/a_1.shape[0]
            rpt_360.append([i,ra_0,ra_1])
        
        #find peaks
        a_360_x=np.array(rpt_360)[:,0]
        a_360_y0=np.array(rpt_360)[:,1]
        a_360_y1=np.array(rpt_360)[:,2]
        idx_min0,idx_max0=turning_points(a_360_y0)
        idx_min1,idx_max1=turning_points(a_360_y1)
        rpt_360_peaks0=[]
        rpt_360_peaks1=[]
        
        for i in idx_max0:
            rpt_360_peaks0.append([a_360_x[i],a_360_y0[i]])    
        for i in idx_max1:
            rpt_360_peaks1.append([a_360_x[i],a_360_y1[i]])
            
        arr0=np.array(rpt_360_peaks0)    
        arr1=np.array(rpt_360_peaks1)
        a_360_peaks0=arr0[arr0[:, 1].argsort()]
        a_360_peaks1=arr1[arr1[:, 1].argsort()]

        
        #plot by deg chart
        plt.figure(figsize=(8,4))
        plt.title('NSS EDGE Image Quality Ra by Deg')
        plt.plot(a_360_x,a_360_y0,label=os.path.basename(img_file) + '(raw)')
        plt.plot(a_360_x,a_360_y1,label=os.path.basename(img_file) + '(Moving Avg/Original Ra)')

        plt.scatter(a_360_peaks0[-10:,0],a_360_peaks0[-10:,1],facecolors='none', edgecolors='r')
        plt.scatter(a_360_peaks0[-20:-10,0],a_360_peaks0[-20:-10,1],facecolors='none', edgecolors='g')
        
        plt.scatter(a_360_peaks1[-10:,0],a_360_peaks1[-10:,1],facecolors='none', edgecolors='r')
        plt.scatter(a_360_peaks1[-20:-10,0],a_360_peaks1[-20:-10,1],facecolors='none', edgecolors='g')

        
        plt.ylim(0,150)
        plt.xticks(np.arange(0, 360, 10.0))
        plt.xticks(rotation=90, ha='left')
        plt.xlabel('Angle NotchDown CW')
        plt.legend()
        plt.savefig(nss_img_path + os.path.basename(f)[:-4] + '_360chart.jpg')
        #plt.show()
        plt.clf()
        plt.close()

        #excluded notch before 2500 and last 2500 pixels
        #=================================================朱sir原始檔案中有輸出，但分析時基本上用不到，故先mark掉，要用時打開就好
        '''
        plt.figure(figsize=(8,4))
        plt.title('NSS EDGE Image Quality Ra_raw=' + str(round(Ra_0,2))+ 'Ra_mv=' + str(round(Ra_1,2)) )
        plt.plot(ax_0[mv_sdev_window_1*3:mv_sdev_window_1*-2],a01_0[mv_sdev_window_1*3:mv_sdev_window_1*-2],label=os.path.basename(img_file) + '(raw)')
        plt.plot(ax_1[mv_sdev_window_1*3:mv_sdev_window_1*-2],a01_1[mv_sdev_window_1*3:mv_sdev_window_1*-2],label=os.path.basename(img_file) + '(Moving Avg/Original Ra)')
        plt.ylim(0,150)
        plt.xlabel('Angle NotchDown CW')
        plt.xticks(np.arange(0, 360, 10.0))
        plt.xticks(rotation=90, ha='left')
        plt.legend()
        plt.savefig(nss_img_path + os.path.basename(f)[:-4] + '_chart.jpg')
        #plt.show()
        plt.clf()
        plt.close()
        '''
        #=================================================朱sir原始檔案中有輸出，但分析時基本上用不到，故先mark掉，要用時打開就好

        #朱sir原始檔案中有輸出每片wafer的csv檔案，但分析時基本上用不到，故先mark掉，要用時打開就好
        #===================================================================================================================
        '''
        #export to save csv files
        np.savetxt(nss_img_path + os.path.basename(f)[:-4] + '_raw_peaks.csv',a_360_peaks0, fmt='%1.3f',delimiter=',')
        np.savetxt(nss_img_path + os.path.basename(f)[:-4] + '_mv_peaks.csv',a_360_peaks1, fmt='%1.3f',delimiter=',')
        np.savetxt(nss_img_path + os.path.basename(f)[:-4] + '_360.csv',np.array(rpt_360), fmt='%1.3f',delimiter=',')
        #np.savetxt(nss_img_path + os.path.basename(f)[:-4] + '.csv',a01, fmt='%1.3f',delimiter=',')
        '''
        #===================================================================================================================

        #save images
        for i in np.arange(mv_sdev_window_1*3,a0_1.shape[0]-mv_sdev_window_1*3):
            x0=int(i)#+519
            x1=int(i+1)#+519
            y0=int(a01_0[i]/100*200)
            y1=int(a01_0[i+1]/100*200)

            x2=int(i)+int(mv_sdev_window_1/2)
            x3=int(i+1)+int(mv_sdev_window_1/2)
            y2=int(a01_1[i]/100*200)
            y3=int(a01_1[i+1]/100*200)
            cv2.line(img_v, (x0, img_v.shape[0]-y0-10), (x1, img_v.shape[0]-y1-10), (255, 255, 255), 1)
            cv2.line(img_v, (x2, img_v.shape[0]-y2-10), (x3, img_v.shape[0]-y3-10), (255, 255, 255), 2) 
        #print(nss_img_path + os.path.basename(f)[:-4] + '.png')
        #save whole wafer image
        cv2.imwrite(nss_img_path + os.path.basename(f)[:-4] + '.png', img_v)

        #crop image for top 20 peaks
        tmp_dir=os.path.dirname(f) + '/' + os.path.basename(f)[:-4]
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        for i in a_360_peaks0[-20:,0]:
            img_v0=img_v[:,int((i-0.3)*mv_sdev_window_1):int((i+1.6)*mv_sdev_window_1)]
            cv2.imwrite(tmp_dir + '/' + str(int(i)) + '.png', img_v0)
        for i in a_360_peaks1[-20:,0]:
            if not os.path.isfile(tmp_dir + '/' + str(int(i)) + '.png'):
                img_v0=img_v[:,int((i-0.3)*mv_sdev_window_1):int((i+1.6)*mv_sdev_window_1)]
                cv2.imwrite(tmp_dir + '/' + str(int(i)) + '.png', img_v0)         

        return [img_file,Ra_0,Q50_0,Q90_0,Q99_0,Ra_1,Q50_1,Q90_1,Q99_1]    
    #else:
    #    print('Execption...')
    #    return []

def decompress():
    # 獲取當前目錄
    mypath = os.path.dirname(os.path.realpath(__file__))
    list_of_all_file=[]
    # 遞迴列出所有7z檔案的絕對路徑
    for root, dirs, files in os.walk(mypath):
        for f in files:
            if f.find('.7z')!=-1:
                fullpath = os.path.join(root, f)
                list_of_all_file.append(fullpath) 

    #依序解壓縮並開一個資料夾存BMP檔案，資料夾檔案名稱為lot+slot位置，檔案名稱為lot+slot位置
    for i in list_of_all_file:
        position=i.index('@')
        waferid=i[position-9:position]+i[position+1:position+3]
        os.mkdir(waferid) #建立資料夾
        archive = py7zr.SevenZipFile(i, mode='r') #讀取7z檔案
        archive.extractall(path=mypath+'/'+waferid) #把讀到的檔案存在新建的資料夾中
        archive.close()
        list_of_all_bmp=[]
        #列出資料夾內bmp檔案的絕對路徑
        for root, dirs, files in os.walk(mypath+'/'+waferid):
            for f in files:
                if f.find('.bmp')!=-1:
                    fullpath = os.path.join(root, f)
                    list_of_all_bmp.append(fullpath)
        #重新命名bmp檔案            
        for j in list_of_all_bmp:
            old_file=j
            new_file=os.path.join(os.path.dirname(j),waferid+'.bmp')
            os.rename(old_file,new_file)
    # 再次遞迴列出所有bmp檔案的絕對路徑
    list_of_all_file=[]
    for root, dirs, files in os.walk(mypath):
        for f in files:
            if f.find('.bmp')!=-1:
                fullpath = os.path.join(root, f)
                list_of_all_file.append(fullpath)
    #移動所有bmp檔案
    for i in list_of_all_file:
        file_source=i
        file_destination=mypath
        shutil.move(file_source, file_destination)

def rename_move_rawdata():
    # 獲取當前目錄
    mypath = os.path.dirname(os.path.realpath(__file__))
    list_of_all_file=[]
    # 遞迴列出所有bmp檔案的絕對路徑
    for root, dirs, files in os.walk(mypath):
        for f in files:
            if f.find('.bmp')!=-1:
                fullpath = os.path.join(root, f)
                list_of_all_file.append(fullpath)
    #重新命名所有bmp檔案
    for i in list_of_all_file:
        position=i.index('@')
        waferid=i[position-9:position]+i[position+1:position+3] 
        old_file=i
        new_file=os.path.join(os.path.dirname(i),waferid+'.bmp')
        os.rename(old_file,new_file)
    # 再次遞迴列出所有bmp檔案的絕對路徑
    list_of_all_file=[]
    for root, dirs, files in os.walk(mypath):
        for f in files:
            if f.find('.bmp')!=-1:
                fullpath = os.path.join(root, f)
                list_of_all_file.append(fullpath)
    #移動所有bmp檔案
    for i in list_of_all_file:
        file_source=i
        file_destination=mypath
        shutil.move(file_source, file_destination)

#===========================以下為執行程式============================================
        
# 1.依序解壓縮並開一個資料夾存BMP檔案，資料夾檔案名稱為lot+slot位置，檔案名稱為lot+slot位置
# 2.將bmp檔案搬到當前檔案夾之下

# def main():
#     #decompress()

#     #獲取當前路徑
#     mypath = os.path.dirname(os.path.realpath(__file__))
#     list_of_all_file=[]
#     # 遞迴列出所有bmp檔案的絕對路徑
#     for root, dirs, files in os.walk(mypath):
#         for f in files:
#             if f.find('.bmp')!=-1:
#                 fullpath = os.path.join(root, f)
#                 list_of_all_file.append(fullpath)
#     summary=[]
#     head=['filename','Ra_raw','RawQ50','RawQ90','RawQ99','Ra_mv','MvQ50','MvQ90','MvQ99']
#     #利用迴圈執行主分析程式 process_bmp
#     for i in list_of_all_file:
#         bmpfile=i
#         summary_list=process_bmp(bmpfile=bmpfile)
#         summary.append(summary_list)
#     #將process_bmp return的raw data存在csv檔案
#     #np.savetxt(fname='nss_image_summary.csv',X=summary,fmt='%s',delimiter=',')

#     #===========================================================================以下為資料整理
#     df=pd.DataFrame(summary, columns=head)
#     #對每個column轉換成浮點數，沒有辦法轉換的文字則該列跳過不轉換
#     for i in df.columns:
#         df[i]=df[i].astype(float,errors='ignore')
#     #輸出excel並調整欄寬
#     name_of_wb='nss_image_summary.xlsx'
#     df.to_excel(name_of_wb, sheet_name='sheet1', index=False)
    
#     # from white_paper_tools import auto_fit 
#     auto_fit(name_of_wb=name_of_wb)

#     os.mkdir('origin_bmp') #建立資料夾給bmp原始檔案日後留存
#     os.mkdir('origin_png') #建立資料夾給png原始檔案日後留存

#     #將原始BMP檔案搬到名為'origin_bmp'的資料夾中，png檔案搬到'origin_png'資料夾

#     result = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]#找到當前資料夾中的檔案名稱和資料夾名稱(無路徑)

#     bmp_list=[i for i in result if fnmatch.fnmatch(i, '*.bmp')] #bmp檔案
#     png_list=[i for i in result if fnmatch.fnmatch(i, '*.png')] #png檔案

#     for i in bmp_list:
#         file_source=mypath+'//'+i
#         file_destination=mypath+'//origin_bmp'
#         shutil.move(file_source, file_destination)

#     for i in png_list:
#         file_source=mypath+'//'+i
#         file_destination=mypath+'//origin_png'
#         shutil.move(file_source, file_destination)

# if __name__=='__main__':
#     main()

# --- CONFIGURATION ---
# 1) Point ROOT_DIR to a server-accessible location.
#    - Linux: a mounted path like /mnt/rawdata
#    - Windows UNC: r"\\temfile300.tem.memc.com\rawdata$"
#    You can also override via env var ROOT_DIR.
ROOT_DIR = Path(
    os.environ.get("ROOT_DIR", r"\\temfile300.tem.memc.com\rawdata$")
).resolve() if os.name != "nt" else Path(
    os.environ.get("ROOT_DIR", r"\\temfile300.tem.memc.com\rawdata$")
)

# Where to extract and process zips (per-run workspace inside ROOT_DIR or a local work dir)
WORKSPACE_BASE = Path(os.environ.get("WORKSPACE_BASE", "./workspace")).resolve()
WORKSPACE_BASE.mkdir(parents=True, exist_ok=True)

# --- HELPERS ---
def safe_join(root: Path, subpath: str) -> Path:
    """Resolve 'subpath' under 'root' and ensure it stays inside (no traversal)."""
    p = (root / subpath).resolve() if os.name != "nt" else (root / subpath)
    # For UNC on Windows, .resolve() may not normalize; do a startswith check when possible.
    root_str = str(root).rstrip("\\/")
    p_str = str(p)
    if not p_str.lower().startswith(root_str.lower()):
        raise ValueError("Access outside of ROOT_DIR is not allowed.")
    return p

def list_dir(path: Path):
    """Return (dirs, zip_files) inside 'path'."""
    dirs, zips = [], []
    try:
        for entry in sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            if entry.is_dir():
                dirs.append(entry)
            elif entry.is_file() and entry.suffix.lower() == ".zip":
                zips.append(entry)
    except PermissionError:
        st.error(f"Permission denied: {path}")
    return dirs, zips

def extract_zip(zip_path: Path, out_dir: Path):
    """Extract a .zip into out_dir; returns extraction root."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir

def find_bmps(root: Path):
    """Find all .bmp files recursively under root."""
    return sorted(root.rglob("*.bmp"))

def present_artifacts(bmp_path: Path):
    """Show likely artifacts created by your analyzer next to the BMP."""
    base = bmp_path.with_suffix("")
    png_whole = base.with_suffix(".png")
    chart_360 = bmp_path.parent / f"{bmp_path.stem}_360chart.jpg"

    imgs = []
    if png_whole.exists():
        imgs.append(("Whole wafer PNG", str(png_whole)))
    if chart_360.exists():
        imgs.append(("360 chart", str(chart_360)))

    if imgs:
        st.subheader(f"Artifacts for {bmp_path.name}")
        for title, p in imgs:
            st.markdown(f"**{title}**")
            st.image(p, use_column_width=True)

# --- UI ---
st.set_page_config(page_title="Server ZIP → BMP Processor", layout="wide")
st.title("Process ZIP (server-side, no upload)")

st.caption(
    "Browse files that already reside on the server. "
    "Pick a **.zip** → extract on server → process contained **.bmp** files with your existing code."
)
st.info(f"ROOT_DIR: `{ROOT_DIR}`")
st.caption("Make sure this path is accessible by the Streamlit process (mounted drive or UNC with permissions).")

# Navigation state
if "relpath" not in st.session_state:
    st.session_state.relpath = ""

# Navigation controls
row1 = st.columns([3, 1, 1])
with row1[0]:
    relpath_input = st.text_input(
        "Browse within ROOT_DIR (relative path):",
        value=st.session_state.relpath,
        help="Use folder buttons to navigate, or type a relative path."
    )
with row1[1]:
    if st.button("Go"):
        st.session_state.relpath = relpath_input
with row1[2]:
    if st.button("Reset to root"):
        st.session_state.relpath = ""

# Resolve current directory safely
try:
    current_dir = safe_join(ROOT_DIR, st.session_state.relpath)
except Exception as e:
    st.error(str(e))
    current_dir = ROOT_DIR
    st.session_state.relpath = ""

st.markdown(f"**Path:** `{current_dir}`")

# Directory listing
if not current_dir.exists() or not current_dir.is_dir():
    st.warning("Directory does not exist or is not a folder.")
    st.stop()

dirs, zips = list_dir(current_dir)

nav_left, nav_right = st.columns(2)
with nav_left:
    st.subheader("Folders")
    if str(current_dir).rstrip("\\/").lower() != str(ROOT_DIR).rstrip("\\/").lower():
        if st.button("⬆️ Up one level"):
            st.session_state.relpath = str(Path(st.session_state.relpath).parent).replace("\\", "/")
            st.rerun()
    for d in dirs:
        if st.button(f"📁 {d.name}", key=f"dir_{d}"):
            new_rel = str(Path(st.session_state.relpath) / d.name).replace("\\", "/")
            st.session_state.relpath = new_rel
            st.rerun()

with nav_right:
    st.subheader("ZIP files")
    if not zips:
        st.info("No .zip files here.")
    else:
        labels = [f.name for f in zips]
        choice = st.selectbox("Select a ZIP:", labels)
        selected_zip = zips[labels.index(choice)]
        st.write(f"**Selected:** `{selected_zip}`")

        # Extraction destination
        default_out = WORKSPACE_BASE / selected_zip.stem
        out_dir = st.text_input(
            "Extraction/output folder (server-side):",
            value=str(default_out),
            help="ZIP will be extracted here; BMPs will be processed in-place."
        )

        col_run = st.columns([1, 1, 2])
        with col_run[0]:
            do_extract = st.button("Extract ZIP")
        with col_run[1]:
            do_process = st.button("Extract + Process BMPs")

        if do_extract or do_process:
            try:
                with st.spinner("Extracting on server..."):
                    out_path = extract_zip(selected_zip, Path(out_dir))
                st.success(f"Extracted to: `{out_path}`")

                if do_process:
                    bmps = find_bmps(out_path)
                    if not bmps:
                        st.warning("No .bmp files found after extraction.")
                    else:
                        st.write(f"Found **{len(bmps)}** BMP files.")
                        progress = st.progress(0)
                        results = []
                        for idx, bmp in enumerate(bmps, start=1):
                            try:
                                res = process_bmp(str(bmp))
                                if isinstance(res, list) and res:
                                    results.append(res)
                                present_artifacts(bmp)
                            except Exception:
                                st.error(f"Failed to process {bmp.name}")
                                st.code(traceback.format_exc())
                            progress.progress(idx / len(bmps))

                        if results:
                            # Expected header from your process_bmp return
                            header = ["filename","Ra_raw","RawQ50","RawQ90","RawQ99","Ra_mv","MvQ50","MvQ90","MvQ99"]
                            # Truncate/align rows defensively
                            rows = []
                            for r in results:
                                rows.append(r[:len(header)] + [None]*max(0, len(header)-len(r)))
                            df = pd.DataFrame(rows, columns=header)
                            st.subheader("Summary")
                            st.dataframe(df)
                            # Optional: save summary beside out_dir
                            try:
                                out_csv = Path(out_dir) / "zip_processing_summary.csv"
                                df.to_csv(out_csv, index=False)
                                st.caption(f"Saved summary: `{out_csv}`")
                            except Exception:
                                pass

            except Exception:
                st.error("Operation failed.")
                st.code(traceback.format_exc())
