import matplotlib
import scipy.fftpack
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import base64
import datetime
import threading
import paho.mqtt.client as paho
import time
import numpy as np
import numpy
from scipy import signal
import csv
import pandas as pd

global radar_counter,xf_final,yf_final,N_final
global a1,a11
global s
global radar_num
radar_num = 2
index1 = 0
a1 = []
a11 = []
s = []
xf_final = []
yf_final = []
N_final = []
radar_counter = []
count_plot = [0, 0]

for k in range(0,(radar_num+1)):
    a1.append({'R1':[],'T1':[],'R2':[],'T2':[],'Initial_time':[]})
    a11.append({'Initial_time':[]})
    s.append(0)
    radar_counter.append(0)

def getFreq(data, lpf_cutoff): #lpf_cutoff
    data = data - np.mean(data)
    N = len(data)
    j = int(len(data)/120)
    ##LOW PASS FILTER
    if (lpf_cutoff!=0):
        w = lpf_cutoff / (j / 2) # Normalize the frequency
        b, a = signal.butter(7, w, 'low')
        data = signal.filtfilt(b, a, data)
    T = 1/j
    y = data
    print("N: ", N)
    yf = scipy.fftpack.fft(y)
    print("fft done")
    yf = yf*(10^5)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    return xf, yf, N

def append_data(x_data, y_data, N_data):
    global xf_final,yf_final,N_final
    print("append data")
    xf_final.append(list(x_data))
    yf_final.append(list(y_data))
    N_final.append(N_data)

def on_connect(client, userdata, flags, rc):

    if rc == 0:

        print("Connected to broker")

        global Connected                #Use global variable
        Connected = True                #Signal connection

    else:

        print("Connection failed")

def plot_figure(xf_final,yf_final,N_final,lpf_cutoff,i,index_temp):
    global a1
    global count_plot
    global radar_counter
    x = radar_counter[index_temp]
    print("plot figure")
    #fig = plt.figure()
    print("index----",index_temp)
    for q in range(i+1):
        
     '''   z = ((2.0/N_final[q])*numpy.abs(yf_final[q][0:N_final[q]//2]))
        
        ## Filter out noise
        threshold = 0.015
        psd_idxs = z > threshold #array of 0 and 1
        psd_clean = z * psd_idxs #zero out all the unnecessary powers
         
        xf_max = (xf_final[q][int(np.where(z==max(z))[0])])
        yf_max = round(np.max(z),1) 
        
        xf_final_temp = xf_final[q]
        xf_final_temp = np.transpose(xf_final_temp) 
        
        Fundamental_freq  = xf_final_temp[(44 < xf_final_temp) & (xf_final_temp < 49)]
        Fundamental_indices = np.where( np.logical_and( xf_final_temp >=44, xf_final_temp <= 49) )[0]
        
        
        FundpeakY = np.max(z[Fundamental_indices]) # Find max peak
        FundlocY = np.argmax(z[Fundamental_indices]) # Find its location
        FundfrqY = Fundamental_freq[FundlocY] # Get the actual frequency value
        
        ThirdHar1 = 3*FundfrqY - 20
        ThirdHar2 = 3*FundfrqY + 20
    
        ThirdHar_freq  = xf_final_temp[(ThirdHar1 < xf_final_temp) & (xf_final_temp < ThirdHar2)]
        ThirdHar_indices = np.where( np.logical_and( xf_final_temp >=ThirdHar1, xf_final_temp <= ThirdHar2) )[0]
         
        ThirdHarpeakY = np.max(z[ThirdHar_indices]) # Find max peak
        ThirdHarlocY = np.argmax(z[ThirdHar_indices]) # Find its location
        ThirdHarfrqY = ThirdHar_freq[ThirdHarlocY] # Get the actual frequency value
        
        TenthHar1 = 10*FundfrqY - 20
        TenthHar2 = 10*FundfrqY + 20

        TenthHar_freq  = xf_final_temp[(TenthHar1 < xf_final_temp) & (xf_final_temp < TenthHar2)]
        TenthHar_indices = np.where( np.logical_and( xf_final_temp >=TenthHar1, xf_final_temp <= TenthHar2) )[0]

        TenthHarpeakY = np.max(z[TenthHar_indices]) # Find max peak
        TenthHarlocY = np.argmax(z[TenthHar_indices]) # Find its location
        TenthHarfrqY = TenthHar_freq[TenthHarlocY] # Get the actual frequency value
        
        PeakRatio = FundpeakY/ThirdHarpeakY
        print("peak ratio",round(PeakRatio,2))
        #plt.figure()
       #ax = plt.subplot(i+1, 1, q+1)
       # ax.plot(xf_final[q], 2.0/N_final[q] * numpy.abs(yf_final[q][:N_final[q]//2]))
        
        ##plt.plot(xf_final, z)
        plt.text(50, yf_max, 'Fund Peak at: '+str(round(float(FundfrqY),2)), fontsize=15)
        plt.text(150, yf_max/2, '3rd Har Peak at: '+str(round(float(ThirdHarfrqY),2)), fontsize=15)
        plt.text(400, yf_max/3, '10th Har Peak at: '+str(round(float(TenthHarfrqY),2)), fontsize=15)
        plt.grid()
        plt.xlabel('Frequency in Hz')
        plt.ylabel('Fourier Amplitude')
        print("Fund freq",round(FundfrqY,2),"Hz ,","3rd Harmonics",round(ThirdHarfrqY,2),"Hz,","10th Harmonics",round(TenthHarfrqY,2),"Hz,","Ratio",round(PeakRatio,2))
        
        #xf_max = (xf_final[q][int(numpy.where(z==max(z))[0])])
        #ax = plt.subplot(i+1, 1, q+1)
        #ax.plot(xf_final[q], 2.0/N_final[q] * numpy.abs(yf_final[q][:N_final[q]//2]))
        #plt.text(xf_max+10, int(max(z)/2), 'Peak at: '+str(round(float(xf_max),2)), fontsize=15)
        fig.suptitle('Radar - ' + str(index_temp), fontsize=20)
        if (lpf_cutoff!=0):
            ax.set_xlim([0, lpf_cutoff+100])

    figManager=plt.get_current_fig_manager()
    figManager.canvas.manager.window.attributes('-topmost', 1)
    figManager.canvas.manager.window.attributes('-topmost', 0)
    #plt.show()
    #fig.savefig('F:/Internship/'+str(index_temp)+'/Radar '+str(index_temp)+'__Plot'+str(count_plot[index_temp-1])+'.png', dpi=fig.dpi)
    #fig.savefig('/home/tcs/rajat/MQTT/PycharmProjects/mqtt_server_client_final/plots/Radar'+str(index_temp)+'/Radar '+str(index_temp)+'__Plot'+str(count_plot)+'.png', dpi=fig.dpi)
    #count_plot +=1
    plt.close()'''
    print("CSV Logging Started")
    ###CSV###
    dict1 = {'Time Stamp1': a1[index_temp]['T1'][x], 'Sample1_data': a1[index_temp]['R1'][x]}
    dict2 = {'Time Stamp2': a1[index_temp]['T2'][x], 'Sample2_data': a1[index_temp]['R2'][x]}
    frame1 = pd.DataFrame(dict1)
    frame2 = pd.DataFrame(dict2)
    cframe = pd.concat([frame1, frame2], axis=1)
    #cframe.to_csv('Radar_log.csv', index=False, columns=["Time Stamp1","Sample1_data","Time Stamp2", "Sample2_data"])
    cframe.to_csv('F:/Internship/'+str(index_temp)+'/Radar_log'+str(count_plot[index_temp-1])+'.csv', index=False, columns=["Time Stamp1","Sample1_data","Time Stamp2", "Sample2_data"])
#cframe.to_csv('/home/tcs/rajat/MQTT/PycharmProjects/mqtt_server_client_final/plots/Radar'
           # +str(index_temp)+'/Radar_log.csv', index=False, columns=["Time Stamp1","Sample1_data","Time Stamp2", "Sample2_data"])
    count_plot[index_temp-1] +=1
    print("::::Logging Finished::::")
    
def on_message(client, userdata, message):
    global xf_final, yf_final, N_final
    global radar_counter
    global a1, a11
    global s
    global index1
    print("inside payload")
    index1 = int(message.topic[-1])
    print("Radar ",index1)
    x1 = str(message.payload.decode("utf-8"))
    s1 = eval(base64.b64decode(x1))
    SampleTime = s1['Initial_time']
    CurrentTimeTemp = datetime.datetime.now()  
    DeltaTime = abs(CurrentTimeTemp - SampleTime)
    DeltaInSeconds = (DeltaTime).total_seconds()
    print("Delta in sec ", DeltaInSeconds)

    if(int(DeltaInSeconds) < 100):
        print("in sec")
        a1[index1]['R1'].append(s1['R1'])
        a1[index1]['T1'].append(s1['T1'])
        a1[index1]['R2'].append(s1['R2'])
        a1[index1]['T2'].append(s1['T2'])
        a1[index1]['Initial_time'].append(s1['Initial_time'])
        r1_data = a1[index1]['R1'][radar_counter[index1]]
        xf, yf, N = getFreq(r1_data,0)
        print("before append")
        append_data(xf,yf,N)
        r2_data = a1[index1]['R2'][radar_counter[index1]]
        xf, yf, N = getFreq(r2_data,0)
        append_data(xf,yf,N)
        plot_figure(xf_final,yf_final,N_final,0,1,index1)
        radar_counter[index1]=radar_counter[index1] + 1
        xf_final = []
        yf_final = []
        N_final = []
    else:
        a11[index1]['Initial_time'].append(s1['Initial_time'])
        print("Rejected Packet Size (Index-" + str(index1) + ") ::"+str(len(a11[index1]['Initial_time'])))

        
Connected = False   #global variable for the state of the connection


broker_address = "192.168.17.66"
client = paho.Client()               
port = 1883
client.on_connect= on_connect                     
client.on_message= on_message                      
client.connect(broker_address,port=port)
client.loop_start()        #start the loop

while Connected != True:    #Wait for connection
    time.sleep(0.1)

client.subscribe("data/radar1")
client.subscribe("data/radar2")
#client.subscribe("data/radar3")

try:
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print ("done")
    client.disconnect()
    client.loop_stop()

