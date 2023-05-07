# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 16:53:10 2021

@author: user
"""

from win32com.client import Dispatch
from os import getcwd
import pandas as pd

Excel=Dispatch("Excel.Application") 
R = {
     'machine':'A1:F2500',
     'pressure':'A1:J2500',
     # 'Temperature':'A5:L3000'
         }
packing_time = ([10]*5 + [11.5]*5 + [13]*5)*3
start = 1
end = 46
root = getcwd()

for DOE in range(start,end):
    root = r'D:\nkustClass\110-碩二上\射出成型實驗室\20220125_KevinData\L45_rebuild(base)\L45' + '\\' + str(DOE)
    data = {}
    for file in ['machine', 'pressure']:
        # print(file)
        filename = root + "\\"+file+".csv"
        # print(filename)
        workbook=Excel.Workbooks.Open(Filename = filename, ReadOnly=1, UpdateLinks=False) 
        data[file] = workbook.Worksheets[file].Range(R[file]).Value
        workbook.Close(False)
    
    machine = [x for x in data['machine'] if None not in x]
    # machine_data = [(x[0],)+x[1::2] for x in machine]
    machine_data = pd.DataFrame(machine[1:], columns=machine[0])


    pressure = [x for x in data['pressure'] if None not in x]
    # pressure_data = [(x[0],)+x[1::2] for x in pressure]
    pressure_data = pd.DataFrame(pressure[1:], columns=pressure[0])
    
    Partweight = max(machine_data['partWeight (g)'])
    machine_time = machine_data['time (sec)']
    pressure_time = pressure_data['time (sec)']
    syspressure = machine_data['injectionPressure (MPa)']
    
    # def find_pt(a,b,c):
    #     if c<b+(b-a)-0.2:
    #         return True
    # pt_idx = [i+1 for i in range(len(syspressure)-2) if find_pt(syspressure[i], syspressure[i+1], syspressure[i+2])][-1]
    pt_time = machine_time[syspressure.argmax()] + packing_time[DOE-start]
    pt_idx = sum(machine_time<pt_time)
    
    def integral(pressure, time_list):
        global pt_idx
        int_pressure = 0
        for i in range(1, pt_idx):
            int_pressure += (time_list[i]-time_list[i-1])*pressure[i]
        return int_pressure
    
    def Residual(pressure, time_list):
        time = 29
        time_idx = sum(time_list<time)
        Residual_pressure = pressure[time_idx-1]+(pressure[time_idx]-pressure[time_idx-1])*(time_list[time_idx]-time)/(time_list[time_idx]-time_list[time_idx-1])
        return(Residual_pressure)
    
    if DOE == start:
        index={}
        index["max_syspressure"] = [max(syspressure)]
        index["int_syspressure"] = [integral(syspressure, machine_time)]
        
        for sn in pressure_data.columns[1:10]:
            print(f'sn: {sn}')
            pressure = pressure_data[sn]
            index['max_'+sn] = [max(pressure)]
            index['int_'+sn] = [integral(pressure, pressure_time)]
            index['res_'+sn] = [Residual(pressure, pressure_time)]
        index["Partweight"] = [Partweight]
    else:
        index["max_syspressure"].append(max(syspressure))
        index["int_syspressure"].append(integral(syspressure, machine_time))
        
        for sn in pressure_data.columns[1:]:
            pressure = pressure_data[sn]
            index['max_'+sn].append(max(pressure))
            index['int_'+sn].append(integral(pressure, pressure_time))
            index['res_'+sn].append(Residual(pressure, pressure_time))
        index["Partweight"].append(Partweight)
            
pd.DataFrame(index).to_csv('L45_index_new.csv')



