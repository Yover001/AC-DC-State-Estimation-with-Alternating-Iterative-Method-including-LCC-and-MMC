import datetime
import numpy as np
from math import pi,sqrt
from numpy import arctan,sin,cos,arccos
from ACSE import ACSE
from DCSE import DCSE
from Polar import AC_Polar,LCC_Polar,VSC_Polar
from Data import GetNodeData,GetLineData,LCC_GetNodeData,VSC_GetNodeData,DC_GetLineData,GetY,GetYdc
from OutTxt import SingleTxt,StringTxt,Real
Out_Path = 'Cal-Process.txt'
starttime = datetime.datetime.now()
#------------------------------------------网络信息读取------------------------------------------------#
# 获取直流系统信息
RootPath = 'C:\\Users\\lenovo\\Desktop\\python代码\\状态估计\\SE-交替\\Data\\'
VSC_Node = RootPath+'VSC_NodeData.txt'
LCC_Node = RootPath+'LCC_NodeData.txt'    
DC_Line = RootPath+'DC_LineData.txt'
VSC_NodeData = VSC_GetNodeData(VSC_Node,show=1)
LCC_NodeData = LCC_GetNodeData(LCC_Node,show=1)
DC_LineData = DC_GetLineData(DC_Line,show=1)   
Ydc=GetYdc(LCC_NodeData,VSC_NodeData,DC_LineData,path=Out_Path,width=6)
#获取交流系统信息
FilePath_Node = RootPath+'NodeData.txt'    
FilePath_Line = RootPath+'LineData.txt'
NodeData = GetNodeData(FilePath_Node,show=1)
LineData = GetLineData(FilePath_Line,show=1)
Y = GetY(NodeData,LineData,path=Out_Path,width=6)
#----------------------------------------------交直流状态估计----------------------------------------#
StringTxt(path=Out_Path,string='交直流状态估计',fmt='w')
S0 = '-'
for i in range(130):
    S0 = S0+'-' 
StringTxt(path=Out_Path,string=S0)
#-初始化   
U,Angle = AC_Polar(NodeData[:,3],NodeData[:,4],path=Out_Path,width=9)
Vd,Id,Kt,W,fi = LCC_Polar(LCC_NodeData,path=Out_Path,width=9) 
Udc,Idc,derta,M = VSC_Polar(VSC_NodeData,path=Out_Path,width=9) 
#-----------------------------交替迭代计算----------------------------------#
L_V = 0
Tol = 1e-6
while True:
	L_V=L_V+1
	print('交替迭代次数：',L_V)
	S0 = '-'
	for i in range(130):
	    S0 = S0+'-' 
	SingleTxt(L_V,path=Out_Path,string='\n交替迭代次数：')
	StringTxt(path=Out_Path,string=S0)
#------------------------------直流状态估计---------------------------------# 
	Iter = 0
	Tol = 1e-6
	MaxIter = 3
	while True:
		Iter = Iter + 1
		Vd,Id,Kt,W,fi,Pss,Qss,Pd,Udc,Idc,derta,M,Pv,Qv,Pdc,Pdcij,MaxError,U_PCC1,D_PCC1= DCSE(U,Angle,Vd,Id,Kt,W,fi,Udc,Idc,derta,M,Ydc,LCC_NodeData,VSC_NodeData,DC_LineData,Tol,path=Out_Path,width=9)
		if Iter>MaxIter or MaxError<Tol:
			break
	# 结束直流循环
	if MaxError<Tol:
		SingleTxt(Iter-1,path=Out_Path,string=S0+'\n直流迭代完成，更新次数为：')
		SingleTxt(MaxError,path=Out_Path,string='直流最大误差为：')
	else:
	    SingleTxt(MaxError,path=Out_Path,string='直流结果不收敛!')
#------------------------------交流状态估计---------------------------------# 	
	Iter = 0
	Tol = 1e-6
	MaxIter = 3
	while True:
		Iter = Iter + 1
		U,Angle,P,Q,Pij,Qij,MaxError,U_PCC2,D_PCC2= ACSE(U,Angle,Y,NodeData,LineData,Pss,Qss,Pv,Qv,LCC_NodeData,VSC_NodeData,Tol,path=Out_Path,width=9)
		if Iter>MaxIter or MaxError<Tol:
			break
        # 结束交流循环
	if MaxError<Tol:
		SingleTxt(Iter-1,path=Out_Path,string=S0+'\n交流迭代完成，更新次数为：')
		SingleTxt(MaxError,path=Out_Path,string='交流最大误差为：')
	else:
		SingleTxt(MaxError,path=Out_Path,string='交流结果不收敛!')
#-------------------------------------------交替迭代收敛判据---------------------------------------------#
	U_MaxError=np.max(np.abs(U_PCC1-U_PCC2))
	D_MaxError=np.max(np.abs(D_PCC1-D_PCC2))
	AD_MaxError=max(U_MaxError,D_MaxError)
	print(AD_MaxError) 
	if L_V>3 or AD_MaxError<Tol:
		break
if AD_MaxError<Tol:
	SingleTxt(L_V,path=Out_Path,string='交替迭代迭代完成，更新次数为：')
	SingleTxt(AD_MaxError,path=Out_Path,string='最大误差为：')
#-------------------------------------------AC状态估计结果---------------------------------------------#
	Real(U,path=Out_Path,string=S0+'\nAC电压：\n')
	Real(P,path=Out_Path,string='注入有功：\n')
	Real(Q,path=Out_Path,string='注入有功：\n')
	Real(Pij,path=Out_Path,string='线路有功：\n')
	Real(Qij,path=Out_Path,string='线路有功：\n')
#-------------------------------------------LCC状态估计结果---------------------------------------------#
	Real(Vd,path=Out_Path,string=S0+'\nLCC直流电压：\n')
	Real(Id,path=Out_Path,string='LCC直流电流：\n')
	Real(Pss,path=Out_Path,string='交流侧有功：\n')
	Real(Qss,path=Out_Path,string='交流侧无功：\n')
	Real(Pd,path=Out_Path,string='直流功率：\n')
#-------------------------------------------VSC状态估计结果---------------------------------------------#
	Real(Udc,path=Out_Path,string=S0+'\nVSC直流电压：\n')
	Real(Idc,path=Out_Path,string='VSC直流电流：\n')
	Real(Pv,path=Out_Path,string='交流侧有功：\n')
	Real(Qv,path=Out_Path,string='交流侧无功：\n')
	Real(Pdc,path=Out_Path,string='直流功率：\n')
	Real(Pdcij,path=Out_Path,string='线路功率：\n')
else:
	SingleTxt(AD_MaxError,path=Out_Path,string='结果不收敛!')  
# TIME
endtime = datetime.datetime.now()
print (endtime - starttime)
# print(Vd,Id,Kt,W,fi,Udc,Idc,derta,M)