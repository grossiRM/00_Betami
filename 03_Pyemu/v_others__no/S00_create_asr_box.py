import flopy
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
modelname = 'ASR_box'
exe = os.path.join('..','gw_codes','mfnwt.exe')
model_ws = os.path.join('ASR_ws')
gis_ws = os.path.join('GIS')

if not os.path.exists(model_ws): os.mkdir(model_ws)
if not os.path.exists(gis_ws): os.mkdir(gis_ws)

mf = flopy.modflow.Modflow(modelname,version='mfnwt',exe_name=exe,model_ws=model_ws)

well_space = 140
delr,delc = 40,40 # start with this assumption
nlay = 6

xbuff = 5280/2
ybuff = 5280/2
Lx,Ly = xbuff + (well_space*5) + xbuff, ybuff + (well_space*5) + ybuff
nrow,ncol = Ly/delc, Lx/delr
if not nrow.is_integer() and ncol.is_integer():
    raise ValueError(f'either nrow: {nrow} or ncol: {ncol} is not an integer')
nrow, ncol = int(nrow),int(ncol)
print(Ly)
lse = 0
z = np.array([600,800,1050,1250,1300,1400])
up_dip_array = np.ones((nlay,1,ncol))
v_slop = 10/5280. # (1 ft per mile)
botm = np.ones((nlay,nrow,ncol))
top = lse
for lay in range(nlay):
     botm[lay][0,:] = up_dip_array[lay] * (lse - z[lay])

for lay in range(nlay):
    for row in range(1,nrow):
        botm[lay][row,:] = botm[lay][0,:] - (row*delc * v_slop)


nper = 21
perlen = [365.25]
steady = [True]
nstp = [1]

for sp in range(0,nper-1):
    perlen.append(365.25)
    steady.append(False)
    nstp.append(1)
laycbd = 0

dis = flopy.modflow.ModflowDis(mf,nlay,nrow,ncol,nper,delr,delc,laycbd,top,botm,perlen,steady=steady)

hk,ss  = [], []
laytyp = []
for lay in range(nlay):
    hk.append(10) #*np.ones((nrow,ncol)))
    ss.append(1e-4) #*np.ones((nrow,ncol)))
    laytyp.append(0)

upw = flopy.modflow.ModflowUpw(mf,laytyp=laytyp,hk=10,ss=1e-4)

spd = {0:[]}
ghb_mask = np.ones((nrow,ncol))
for lay in range(nlay):
    for row in range(1,nrow-1):
        stage = (lse-50) - (row*v_slop*delc)
        spd[0].append([lay,row,0,stage,1000])
        spd[0].append([lay,row,ncol-1,stage,1000])
        ghb_mask[row,0] = stage
        ghb_mask[row,ncol-1] = stage
    for col in range(ncol):
        stage = (lse-50) - (nrow*v_slop*delr)
        spd[0].append([lay,0,col,lse-50,1000])
        spd[0].append([lay,nrow-1,col,stage,1000])

        ghb_mask[0,col] = lse-50
        ghb_mask[nrow-1,col] = stage

ghb = flopy.modflow.ModflowGhb(mf,ipakcb=53,stress_period_data=spd)


# ghb.plot()

strt = top - 50 #[top-50] + (lse - z[:-1]).tolist()
print(strt)
bas = flopy.modflow.ModflowBas(mf,ibound=1,strt=strt)




spd = {}
for i in range(nper):
    spd[(i,0)] = ['save head', 'save budget']
oc = flopy.modflow.ModflowOc(mf,stress_period_data=spd,compact=True) #,save_every=True)


nwt = flopy.modflow.ModflowNwt(mf,maxiterout=5000,linmeth=2,iprnwt=1)

if os.path.exists(os.path.join(model_ws,modelname+'.ghb')):
    os.remove(os.path.join(model_ws,modelname+'.ghb'))
    print('it has been done my lord')

mf.write_input()

mf.run_model()


fig, ax = plt.subplots()
plt.imshow(ghb_mask,cmap='jet')
plt.colorbar()




plt.show()