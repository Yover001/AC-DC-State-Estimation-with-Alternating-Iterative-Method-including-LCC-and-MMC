import math
import OutTxt
import numpy as np
import scipy.linalg
from numpy.linalg import inv
from OutTxt import Real
from math import pi,sqrt
from numpy import arctan,sin,cos,tan,arccos
def DCSE(U,Angle,Vd,Id,Kt,W,fi,Udc,Idc,derta,M,Ydc,LCC_NodeData,VSC_NodeData,DC_LineData,Tol,**Option):
#------------------------------------------LCC不平衡量----------------------------------------#
    AC_LCC=LCC_NodeData[:,0]                            #相连的交流节点(LCC)
    AC_LCC=AC_LCC.astype(int)-1
    AC_VSC=VSC_NodeData[:,0]                            #相连的交流节点(VSC)
    AC_VSC=AC_VSC.astype(int)-1
    Nc = LCC_NodeData.shape[0]
    kr = 0.995
    Vt = np.zeros([Nc,1])
    Dt = np.zeros([Nc,1])                      # 交流电压
    for i in range(Nc):
        Vt[i] = U[AC_LCC[i]]
        Dt[i] = Angle[AC_LCC[i]]
    Uc=np.append(Vd,Udc)                       # LCC 与 MMC 直流电压
    N_LCC = LCC_NodeData[:,12]                 # 换流器组数
    N_VSC = VSC_NodeData[:,12]                 
    N = np.append(N_LCC,N_VSC)
    X = LCC_NodeData[:,5]                  #换相电抗
    Xc= LCC_NodeData[:,6]                  #无功补偿电纳 注意是电纳
    Pss = np.zeros([Nc,1])
    Qss = np.zeros([Nc,1])
    Pd = np.zeros([Nc,1])
    # 控制参数——伪量测
    Kontrol1 = LCC_NodeData[:,3] 
    Kontrol2 = LCC_NodeData[:,4]               
    #量测值
    Psm = LCC_NodeData[:,13]
    Qsm = LCC_NodeData[:,14]
    Pdm = LCC_NodeData[:,15]
    VdM = LCC_NodeData[:,16]
    IdM = LCC_NodeData[:,17] 
    # 10个量测方程
    Delta_D1 = np.zeros([Nc,1])   # 交流侧有功量测             
    Delta_D2 = np.zeros([Nc,1])   # 交流侧无功量测 
    Delta_D3 = np.zeros([Nc,1])   # 直流有功量测
    Delta_D4 = np.zeros([Nc,1])   # 直流电压量测
    Delta_D5 = np.zeros([Nc,1])   # 直流电流量测
    Delta_D6 = np.zeros([Nc,1])   # 直流电压伪量测            
    Delta_D7 = np.zeros([Nc,1])   # 直流电压伪量测
    Delta_D8 = np.zeros([Nc,1])   # 直流网络方程伪量测
    Delta_D9 = np.zeros([Nc,1])   # 控制方程伪量测
    Delta_D10 = np.zeros([Nc,1])  # 控制方程伪量测

    for i in range(Nc):                        # 计算Delta_
        Pss[i] = N_LCC[i]*Vd[i]*Id[i]
        Qss[i] = N_LCC[i]*Vd[i]*Id[i]*np.tan(fi[i])
        # s-Vt[i]*Vt[i]*Xc #注意
        Pd[i] = N_LCC[i]*Vd[i]*Id[i]
        Delta_D1[i] = Psm[i]-Pss[i] 
        Delta_D2[i] = Qsm[i]-Qss[i]
        Delta_D3[i] = Pdm[i]-Pd[i]
        Delta_D4[i] = VdM[i]-Vd[i]
        Delta_D5[i] = IdM[i]-Id[i]
        Delta_D6[i] = Vd[i]-2.7*Kt[i]*Vt[i]*W[i]+1.9*X[i]*Id[i] # 伪量测
        Delta_D7[i] = Vd[i]-2.7*kr*Kt[i]*Vt[i]*cos(fi[i])
        if LCC_NodeData[i,2]==1:  # 定电流 定控制角
            Delta_D8[i] = Id[i]-np.sum(Ydc[i,:]*Uc*N)
            Delta_D9[i] =Id[i]-Kontrol1[i]
            Delta_D10[i] = W[i]-cos(Kontrol2[i])       
        else:                     # 定电压 定控制角
            Delta_D8[i] = -Id[i]-np.sum(Ydc[i,:]*Uc*N)
            Delta_D9[i] =Vd[i]-Kontrol1[i]
            Delta_D10[i] = W[i]-cos(Kontrol2[i])            
#------------------------------------------VSC不平衡量----------------------------------------#
    VSC_Num = VSC_NodeData.shape[0]
    R = VSC_NodeData[:,6]
    Xl = VSC_NodeData[:,7]
    a = arctan(R/Xl)
    Y = 1/np.sqrt(R*R+Xl*Xl)
    Usi = np.zeros([VSC_Num,1])
    Dsi = np.zeros([VSC_Num,1])
    for i in range(VSC_Num):
        Usi[i] = U[AC_VSC[i]]
        Dsi[i] = Angle[AC_VSC[i]]                 # 交流母线电压 
    Pv = np.zeros([VSC_Num,1])
    Qv = np.zeros([VSC_Num,1])
    Pdc = np.zeros([VSC_Num,1])
    PM = VSC_NodeData[:,13]#量测值
    QM = VSC_NodeData[:,14]
    PdcM = VSC_NodeData[:,15]
    UM = VSC_NodeData[:,16]
    IM = VSC_NodeData[:,17]
    #伪量测 
    Pcontronl = VSC_NodeData[:,3]                     # 换流器控制值(迭代过程中不变)
    Ucontronl = VSC_NodeData[:,8] 
    Qcontronl = VSC_NodeData[:,4]
    #-----8量测-----#
    Deltad1 = np.zeros([VSC_Num,1])   # 交流侧有功量测
    Deltad2 = np.zeros([VSC_Num,1])   # 交流侧无功量测
    Deltad3 = np.zeros([VSC_Num,1])   # 直流功率量测
    Deltad4 = np.zeros([VSC_Num,1])   # 直流网络方程伪量测
    Deltad5 = np.zeros([VSC_Num,1])   # 直流电压量测
    Deltad6 = np.zeros([VSC_Num,1])   # 直流电流量测
    Deltad7 = np.zeros([VSC_Num,1])   # 控制方程伪量测
    Deltad8 = np.zeros([VSC_Num,1])   # 控制方程伪量测
    iter = 0 
    for i in range(VSC_Num):                   # 求解功率不平衡量
        Pv[i] =  (sqrt(6)/4)*N_VSC[i]*M[i]*Usi[i]*Udc[i]*Y[i]*sin(derta[i]-a[i]) + N_VSC[i]*Usi[i]*Usi[i]*Y[i]*sin(a[i])
        Qv[i] = -(sqrt(6)/4)*N_VSC[i]*M[i]*Usi[i]*Udc[i]*Y[i]*cos(derta[i]-a[i]) + N_VSC[i]*Usi[i]*Usi[i]*Y[i]*cos(a[i])                     
        Pdc[i] = N_VSC[i]*Udc[i]*Idc[i]
        Deltad1[iter] = PM[i]-Pv[i] 
        Deltad2[iter] = QM[i]-Qv[i] 
        Deltad3[iter] = PdcM[i]-Pdc[i] 
        Deltad4[iter] = Idc[i]-np.sum((Ydc[i+Nc,:]*Uc*N))
        Deltad5[iter] = UM[i]- Udc[i] 
        Deltad6[iter] = IM[i]- Idc[i]
         #--------1-PQ,2-UQ-----------#
        if  VSC_NodeData[i,2]==1:
            Deltad7[iter] = Pv[i]-Pcontronl[i] 
            Deltad8[iter] = Qv[i]-Qcontronl[i]
        else:
            Deltad7[iter] = Udc[i]-Ucontronl[i] 
            Deltad8[iter] = Qv[i]-Qcontronl[i]
        iter = iter+1
    #  直流线路功率量测
    n_DC=Ydc.shape[0]
    n_Line=DC_LineData.shape[0] 
    xdc= [int(i) for i in DC_LineData[:,0]]
    ydc= [int(i) for i in DC_LineData[:,1]]
    Pijdc_Real= DC_LineData[:,3]
    Pijdc = np.zeros([n_Line,1])
    DeltaPijdc=np.zeros([n_Line,1])
    for i in range(n_Line):
        q = xdc[i]-1
        r = ydc[i]-1
        Pijdc[i]=-N[q]*Uc[q]*Ydc[q,r]*(N[q]*Uc[q]-N[r]*Uc[r])
        DeltaPijdc[i] = Pijdc_Real[i]-Pijdc[i]  
    DeltaD = np.vstack([Delta_D1,Delta_D2,Delta_D3,Delta_D4,Delta_D5,Delta_D6,Delta_D7,Delta_D8,Delta_D9,Delta_D10,Deltad1,Deltad2,Deltad3,Deltad4,Deltad5,Deltad6,Deltad7,Deltad8,DeltaPijdc])  # 不平衡量
    # Option['string'] = '不平衡量为：\n'
    # Real(DeltaD,**Option)
    # MaxError = np.max(np.abs(DeltaD)) 
#------------------------------------------------LCC------------------------------------------------#   
    # 状态变量：Vd Id Kt W fi
    #Psm
    F11 = -np.diag(N_LCC*Id)
    F12 = -np.diag(N_LCC*Vd)
    F13 = np.zeros([Nc,Nc])
    F14 = np.zeros([Nc,Nc])
    F15 = np.zeros([Nc,Nc])
    #Qsm
    F21 = -np.diag(N_LCC*Id*tan(fi.reshape(Nc)))
    F22 = -np.diag(N_LCC*Vd*tan(fi.reshape(Nc)))
    F23 = np.zeros([Nc,Nc])
    F24 = np.zeros([Nc,Nc])
    F25 = -np.diag(N_LCC*Vd*Id*(1/cos(fi.reshape(Nc)))*(1/cos(fi.reshape(Nc))))
    #Pdm
    F31 = -np.diag(N_LCC*Id)
    F32 = -np.diag(N_LCC*Vd)
    F33 = np.zeros([Nc,Nc])
    F34 = np.zeros([Nc,Nc])
    F35 = np.zeros([Nc,Nc])
    #Vdm
    F41 = -np.eye(Nc)
    F42 = np.zeros([Nc,Nc])
    F43 = np.zeros([Nc,Nc])
    F44 = np.zeros([Nc,Nc])
    F45 = np.zeros([Nc,Nc])
    #Idm
    F51 = np.zeros([Nc,Nc])
    F52 = -np.eye(Nc)
    F53 = np.zeros([Nc,Nc])
    F54 = np.zeros([Nc,Nc])
    F55 = np.zeros([Nc,Nc])
    #电压wei量测
    F61 = np.eye(Nc)
    F62 = np.diag(1.9*X)
    F63 = -np.diag(Vt.reshape(Nc)*W*2.7)
    F64 = -np.diag(Kt*Vt.reshape(Nc)*2.7)
    F65 = np.zeros([Nc,Nc])
    F71 = np.eye(Nc)
    F72 = np.zeros([Nc,Nc])
    F73 = -np.diag(kr*Vt.reshape(Nc)*cos(fi.reshape(Nc))*2.7)
    F74 = np.zeros([Nc,Nc])
    F75 = np.diag(kr*Kt*Vt.reshape(Nc)*sin(fi.reshape(Nc))*2.7)
    #网络方程wei量测
    F81 = np.zeros([Nc,Nc])
    F82 = np.eye(Nc)
    for i in range(Nc):
        if LCC_NodeData[i,2]==2:
            F82[i,i]=F82[i,i]*(-1)
    F83 = np.zeros([Nc,Nc])
    F84 = np.zeros([Nc,Nc])
    F85 = np.zeros([Nc,Nc])
    #控制伪量测
    F91 = np.zeros([Nc,Nc])
    F92 = np.zeros([Nc,Nc])
    for i in range(Nc):
        if LCC_NodeData[i,2]==2:
            F91[i,i]=1  #定电压
        else:             
            F92[i,i]=1  #定电流
    F93 = np.zeros([Nc,Nc])
    F94 = np.zeros([Nc,Nc])
    F95 = np.zeros([Nc,Nc])
    F101 = np.zeros([Nc,Nc])
    F102 = np.zeros([Nc,Nc])
    F103 = np.zeros([Nc,Nc])
    F104 = np.eye(Nc)  #控制角
    F105 = np.zeros([Nc,Nc])
    F = np.vstack([np.hstack([F11,F12,F13,F14,F15]),np.hstack([F21,F22,F23,F24,F25]),np.hstack([F31,F32,F33,F34,F35]),np.hstack([F41,F42,F43,F44,F45]),np.hstack([F51,F52,F53,F54,F55]),np.hstack([F61,F62,F63,F64,F65]),np.hstack([F71,F72,F73,F74,F75]),np.hstack([F81,F82,F83,F84,F85]),np.hstack([F91,F92,F93,F94,F95]),np.hstack([F101,F102,F103,F104,F105])]) 
#----------------------------------------------VSC-------------------------------------------------#
    # 状态变量：Udc Idc derta M
    # Psm
    D11=np.zeros([VSC_Num,VSC_Num])
    D12=np.zeros([VSC_Num,VSC_Num]) 
    D13=np.zeros([VSC_Num,VSC_Num])
    D14=np.zeros([VSC_Num,VSC_Num])
    # Qsm
    D21=np.zeros([VSC_Num,VSC_Num])
    D22=np.zeros([VSC_Num,VSC_Num]) 
    D23=np.zeros([VSC_Num,VSC_Num])
    D24=np.zeros([VSC_Num,VSC_Num])
    # Pdm
    D31=np.zeros([VSC_Num,VSC_Num])
    D32=np.zeros([VSC_Num,VSC_Num]) 
    D33=np.zeros([VSC_Num,VSC_Num])
    D34=np.zeros([VSC_Num,VSC_Num])
    # 直流网络方程伪量测
    D41=np.zeros([VSC_Num,VSC_Num])
    D42=np.eye(VSC_Num)             
    D43=np.zeros([VSC_Num,VSC_Num]) 
    D44=np.zeros([VSC_Num,VSC_Num])
    # Vdm
    D51=-np.eye(VSC_Num)   # 单位阵
    D52=np.zeros([VSC_Num,VSC_Num]) # 0
    D53=np.zeros([VSC_Num,VSC_Num]) # 0阵 
    D54=np.zeros([VSC_Num,VSC_Num]) # 0
    # Idm
    D61=np.zeros([VSC_Num,VSC_Num])
    D62=-np.eye(VSC_Num)   # 单位阵 
    D63=np.zeros([VSC_Num,VSC_Num]) # 0阵 
    D64=np.zeros([VSC_Num,VSC_Num]) # 0
    #------P or U------#
    D71=np.zeros([VSC_Num,VSC_Num])
    D72=np.zeros([VSC_Num,VSC_Num])
    D73=np.zeros([VSC_Num,VSC_Num]) # 0阵 
    D74=np.zeros([VSC_Num,VSC_Num]) # 0
    #------Q------#
    D81=np.zeros([VSC_Num,VSC_Num])
    D82=np.zeros([VSC_Num,VSC_Num]) # 0
    D83=np.zeros([VSC_Num,VSC_Num]) # 0阵 
    D84=np.zeros([VSC_Num,VSC_Num]) # 0 
    for i in range(VSC_Num):
        D11[i,i]=-(sqrt(6)/4)*N_VSC[i]*M[i]*Usi[i]*Y[i]*sin(derta[i]-a[i])
        D13[i,i]=-(sqrt(6)/4)*N_VSC[i]*M[i]*Usi[i]*Udc[i]*Y[i]*cos(derta[i]-a[i])
        D14[i,i]=-(sqrt(6)/4)*N_VSC[i]*Usi[i]*Udc[i]*Y[i]*sin(derta[i]-a[i])
        D21[i,i]=(sqrt(6)/4)*N_VSC[i]*M[i]*Usi[i]*Y[i]*cos(derta[i]-a[i])
        D23[i,i]=-(sqrt(6)/4)*N_VSC[i]*M[i]*Usi[i]*Udc[i]*Y[i]*sin(derta[i]-a[i])
        D24[i,i]=(sqrt(6)/4)*N_VSC[i]*Usi[i]*Udc[i]*Y[i]*cos(derta[i]-a[i])
        D31[i,i]=-N_VSC[i]*Idc[i]
        D32[i,i]=-N_VSC[i]*Udc[i]
        if VSC_NodeData[i,2]==2:
            D71[i,i]=1
    D = np.vstack([np.hstack([D11,D12,D13,D14]),np.hstack([D21,D22,D23,D24]),np.hstack([D31,D32,D33,D34]),np.hstack([D41,D42,D43,D44]),np.hstack([D51,D52,D53,D54]),np.hstack([D61,D62,D63,D64]),np.hstack([D71,D72,D73,D74]),np.hstack([D81,D82,D83,D84])])      
#-----------------------------------------------整合雅克比矩阵---------------------------------------#
    J_D = scipy.linalg.block_diag(F,D)
    #   补充直流线路功率偏导
    DCline_tracker = np.array([xdc,ydc])    
    Pdcij=np.zeros([n_Line,5*Nc+4*VSC_Num])
    for i in range(n_Line):
        temp = DCline_tracker[:,i]
        temp[0] -= 1
        temp[1] -= 1 
        for j in range(Nc):
            if j == temp[0]:    
                Pdcij[i,j] = -(-2*N[temp[0]]*N[temp[0]]*Uc[temp[0]]+N[temp[1]]*N[temp[0]]*Uc[temp[1]])*Ydc[temp[0],temp[1]]  #   VSC
            if j == temp[1]:
                Pdcij[i,j] = -N[temp[0]]*N[temp[1]]*Uc[temp[0]]*Ydc[temp[0],temp[1]]
        for j in range(VSC_Num):
            if j == (temp[0]-Nc):    
                Pdcij[i,5*Nc+j] = -(-2*N[temp[0]]*N[temp[0]]*Uc[temp[0]]+N[temp[1]]*N[temp[0]]*Uc[temp[1]])*Ydc[temp[0],temp[1]]  #   VSC
            if j == (temp[1]-Nc):
                Pdcij[i,5*Nc+j] = -N[temp[0]]*N[temp[1]]*Uc[temp[0]]*Ydc[temp[0],temp[1]]
    J_J = np.vstack([J_D,Pdcij])
#-------------------------------------------------求解----------------------------------------------#
    # Option['string'] = 'jacobi矩阵为：\n'
    # Real(J_J,**Option)
    Gain = J_J.T.dot(J_J)
    # Gain
    Gain[0,0] = 10000000
    niag = inv(Gain)
    Delta = inv(Gain).dot(J_J.T).dot(DeltaD)
    # print(DeltaD)
    # Option['string'] = '方程组求解结果：\n'
    # Real(Delta,**Option)
#-------------------------------------------------修正------------------------------------------#
    Vd = Vd-Delta[0:Nc].reshape(Nc)                                             
    Id = Id-Delta[Nc:2*Nc].reshape(Nc)
    Kt = Kt-Delta[2*Nc:3*Nc].reshape(Nc)
    W= W-Delta[3*Nc:4*Nc].reshape(Nc)
    fi = fi-Delta[4*Nc:5*Nc]
    Udc = Udc-Delta[5*Nc:5*Nc+VSC_Num].reshape(VSC_Num)         
    Idc = Idc-Delta[5*Nc+VSC_Num:5*Nc+2*VSC_Num].reshape(VSC_Num)
    derta = derta-Delta[5*Nc+2*VSC_Num:5*Nc+3*VSC_Num].reshape(VSC_Num)
    M = M-Delta[5*Nc+3*VSC_Num:5*Nc+4*VSC_Num].reshape(VSC_Num)
    # Option['string'] = '\nLCC更新之后的直流电压为：\n'
    # Real(Vd,**Option)
    # Option['string'] = '直流电流为：\n'
    # Real(Id,**Option)
    # Option['string'] = '换流变变比：\n'
    # Real(Kt,**Option)
    # Option['string'] = '控制角\n：'
    # Real(57.3*arccos(W),**Option)
    # Option['string'] = '功率因数：\n'
    # Real(cos(fi),**Option)
    # Option['string'] = '\nVSC更新之后的直流电压为：\n'
    # Real(Udc,**Option)
    # Option['string'] = '直流电流为：\n'
    # Real(Idc,**Option)
    # # Option['string'] = '功角：\n'
    # Real(57.3*derta,**Option)
    # Option['string'] = '调制比：\n'
    # Real(M,**Option)
    MaxError = np.max(np.abs(Delta))
    U_PCC=np.append(Vt,Usi)
    D_PCC=np.append(Dt,Dsi)
    return(Vd,Id,Kt,W,fi,Pss,Qss,Pd,Udc,Idc,derta,M,Pv,Qv,Pdc,Pijdc,MaxError,U_PCC,D_PCC)