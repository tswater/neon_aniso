import numpy as np
import h5py
import pickle
import os
import inspect

sites=[]
for file in os.listdir('/home/tswater/Documents/Elements_Temp/NEON/neon_1m/'):
    if 'L1' in file:
        sites.append(file[0:4])
sites.sort()

vmx_a=.7
vmn_a=.1

anibins=np.linspace(vmn_a,vmx_a,11)
zLbins=np.logspace(-4,2,40)[-1:0:-1]

##### FUNCITONS #####
def skill_old(d_0,d_old,d_new):
    return 1-np.nanmedian(np.abs(d_0-d_new))/np.median(np.abs(d_0-d_old))

def skill(d_0,d_old,d_new):
    mad_n=np.nanmedian(np.abs(d_0-d_new))
    mad_o=np.median(np.abs(d_0-d_old))
    ss=1-np.nanmedian(np.abs(d_0-d_new))/np.median(np.abs(d_0-d_old))
    return mad_n,mad_o,ss

def add_species_old(d_,sp,phi,zL,ani,phi_new,phi_old,fpsite,stb=False):
    d_[sp]['phi']=float('nan')
    d_[sp]['zL']=float('nan')
    d_[sp]['ani']=float('nan')

    mlo=np.abs(zL)<.1
    mhi=np.abs(zL)>.1

    d_[sp]['SS']=skill(phi,phi_old,phi_new)
    d_[sp]['SSlo']=skill(phi[mlo],phi_old[mlo],phi_new[mlo])
    d_[sp]['SShi']=skill(phi[mhi],phi_old[mhi],phi_new[mhi])
    d_[sp]['SS_s']={}
    d_[sp]['SSlo_s']={}
    d_[sp]['SShi_s']={}

    for site in sites:
        mhis=mhi.copy()
        mlos=mlo.copy()
        mlos=mlos&((site.encode('UTF-8'))==fpsite)
        mhis=mhis&((site.encode('UTF-8'))==fpsite)
        ms=((site.encode('UTF-8'))==fpsite)
        d_[sp]['SS_s'][site]=skill(phi[ms],phi_old[ms],phi_new[ms])
        d_[sp]['SSlo_s'][site]=skill(phi[mlos],phi_old[mlos],phi_new[mlos])
        d_[sp]['SShi_s'][site]=skill(phi[mhis],phi_old[mhis],phi_new[mhis])

    return d_[sp]

def add_species(d_,sp,phi,zL,ani,phi_new,phi_old,fpsite,stb=False):
    mlo=np.abs(zL)<.1
    mhi=np.abs(zL)>.1

    d_[sp]['MAD_SC23'],d_[sp]['MAD_OLD'],d_[sp]['SS']=skill(phi,phi_old,phi_new)
    d_[sp]['MAD_SC23lo'],d_[sp]['MAD_OLDlo'],d_[sp]['SSlo']=skill(phi[mlo],phi_old[mlo],phi_new[mlo])
    d_[sp]['MAD_SC23hi'],d_[sp]['MAD_OLDhi'],d_[sp]['SShi']=skill(phi[mhi],phi_old[mhi],phi_new[mhi])
    d_[sp]['MAD_SC23_s']={}
    d_[sp]['MAD_SC23lo_s']={}
    d_[sp]['MAD_SC23hi_s']={}
    d_[sp]['MAD_OLD_s']={}
    d_[sp]['MAD_OLDlo_s']={}
    d_[sp]['MAD_OLDhi_s']={}
    d_[sp]['SS_s']={}
    d_[sp]['SSlo_s']={}
    d_[sp]['SShi_s']={}

    for site in sites:
        mhis=mhi.copy()
        mlos=mlo.copy()
        mlos=mlos&((site.encode('UTF-8'))==fpsite)
        mhis=mhis&((site.encode('UTF-8'))==fpsite)
        ms=((site.encode('UTF-8'))==fpsite)
        d_[sp]['MAD_SC23_s'][site],d_[sp]['MAD_OLD_s'][site],d_[sp]['SS_s'][site]=skill(phi[ms],phi_old[ms],phi_new[ms])
        d_[sp]['MAD_SC23lo_s'][site],d_[sp]['MAD_OLDlo_s'][site],d_[sp]['SSlo_s'][site]=skill(phi[mlos],phi_old[mlos],phi_new[mlos])
        d_[sp]['MAD_SC23hi_s'][site],d_[sp]['MAD_OLDhi_s'][site],d_[sp]['SShi_s'][site]=skill(phi[mhis],phi_old[mhis],phi_new[mhis])

    return d_[sp]


def get_phi(fp,var):
    zL_=(fp['zzd'][:])/fp['L_MOST'][:]
    ani=np.sqrt(fp['WW'][:])/fp['Ustr'][:]#fp['ANI_YB'][:]
    if 'U' in var:
        phi=np.sqrt(fp['UU'][:])/fp['USTAR'][:]
    elif 'V' in var:
        phi=np.sqrt(fp['VV'][:])/fp['USTAR'][:]
    elif 'W' in var:
        phi=np.sqrt(fp['WW'][:])/fp['USTAR'][:]
    elif 'T' in var:
        phi=np.abs(fp['T_SONIC_SIGMA'][:]/(fp['WTHETA'][:]/fp['USTAR'][:]))
    elif 'H2O' in var:
        molh2o=18.02*10**(-3)
        moldry=28.97*10**(-3)
        kgdry_m3=fp['RHO'][:]/(fp['H2O'][:]/1000+1)
        rr = fp['H2O_SIGMA'][:]*molh2o/moldry
        lhv=2500827 - 2360*(fp['T_SONIC'][:]-273)
        phi=np.abs(rr/(fp['LE'][:]/lhv/fp['USTAR'][:]))*kgdry_m3/10**3
    elif 'CO2' in var:
        molh2o=18.02*10**(-3)
        moldry=28.97*10**(-3)
        co2 = fp['CO2_SIGMA'][:]
        if 's' in var:
            co2=co2/10**6
        kgdry_m3=fp['RHO'][:]/(fp['H2O'][:]/1000+1)
        moldry_m3=kgdry_m3/moldry
        phi=np.abs(co2/(fp['CO2FX'][:]/fp['USTAR'][:]))*moldry_m3
    return phi, ani, zL_


#########################

##### CURVE FUNCTIONS #####
def old(var,zL):
    if var=='Uu':
        return 2.55*(1-3*zL)**(1/3)
    elif var=='Vu':
        return 2.05*(1-3*zL)**(1/3)
    elif var=='Wu':
        return 1.35*(1-3*zL)**(1/3)
    elif var=='Tu':
        phi_old=.99*(.067-zL)**(-1/3)
        phi_old[zL>-0.05]=.015*(-zL[zL>-0.05])**(-1)+1.76
        return phi_old
    elif var=='H2Ou':
        return np.sqrt(30)*(1-25*zL)**(-1/3)
    elif var=='CO2u':
        return np.sqrt(30)*(1-25*zL)**(-1/3)
    elif var=='Us':
        return 2.06*np.ones((len(zL)))
    elif var=='Vs':
        return 2.06*np.ones((len(zL)))
    elif var=='Ws':
        return 1.6*np.ones((len(zL)))
    elif var=='Ts':
        return 0.00087*(zL)**(-1.4)+2.03
    elif var=='H2Os':
        return 2.74*np.ones((len(zL)))
    elif var=='CO2s':
        return 2.74*np.ones((len(zL)))
    else:
        return float('nan')

#def T_stb(zL,a,b,c,d):
#    logphi=a+b*np.log10(zL)+c*np.log10(zL)**2+d*np.log10(zL)**3
#    phi=10**(logphi)
#    return phi
def T_stb(zL,a,b):
    return a*(zL)**(-1.4)+b #sfyri reference

def T_ust(zL,a):
    phi=1.07*(0.05+np.abs(zL))**(-1/3)+\
        (-1.14+a*(np.abs(zL)**(-9/10)))*(1-np.tanh(10*(np.abs(zL))**(2/3)))
    return phi

def T_ust2(zL,a):
    phi=1.07*(0.05+np.abs(zL))**(-1/3)+\
        (-1.14+a*(np.abs(zL)**(-9/10)))*(1-np.tanh(10*(np.abs(zL))**(2/3)))
    return phi

def U_stb(zL,a,c):
    phi=a*(1+3*zL)**(c)
    return phi

def U_ust(zL,a):
    return a*(1-3*zL)**(1/3)

def W_stb(zL,a,c):
    return U_stb(zL,a,c)

def W_ust(zL,a):
    return U_ust(zL,a)

def C_stb(zL,a,b):
    return a*(zL)**(-.05)+b #Pahlow 2001

def C_stb2(zL,a,b):
    return a*(zL)**(-.15)+b #Pahlow 2001

def C_ust(zL,a):
    #return a*(1+b*zL)**(-1/3)
    return a*(1-25*zL)**(-1/3)

#####################

prms={'Uu':{'a':[ 3.89679961 ,-9.13245203,  7.63487022]},
'Vu':{'a':[ 3.70189148, -7.05659993, 4.53410038]},
'Wu':{'a':[  0.63163208 ,  3.88577255 ,-10.12068558  , 7.72460201]},
'Tu':{'a':[ 0.28648304, -0.30780311]},
'H2Ou':{'a':[ 14.37651354 ,-27.1532022 ,  25.37032234]},
'CO2u':{'a':[  44.57606848, -123.89935689 , 102.02258138]},
'Us':{'a':[ 1.6802681 , -0.83479918 , 2.78383377],'b':[ 0.23025368 ,-0.21979446 , 0.10754936]},
'Vs':{'a':[ 2.46798009, -2.85310554 , 3.86412966],'b':[ 0.11880141 , 0.11264626 ,-0.01234612 ,-0.20203458]},
'Ws':{'a':[0.87219649, 0.29881858, 2.4331095 ],'b':[ 0.11125322,  0.13588297, -0.18262059,  0.00956019]},
'Ts':{'a':[ 0.55393087, -0.60651883],'b':[-0.03668198 , 0.56982596, -1.30473498,  0.71352531],'c':[ 0.01669026, -0.36187085,  0.75704627, -0.2980538 ],'d':[-0.01921785, -0.05521787,  0.04072313]},
'H2Os':{'a':[ 0.68477546, -0.53585498],'b':[-0.07049873,  0.73430873, -2.26273916,  1.76548149],'c':[-0.02677078,  0.33829338, -1.30150329 , 1.27311958],'d':[-0.00562843 , 0.05189838, -0.22204314 , 0.23390892]},
'CO2s':{'a':[ 0.61316064, -0.51797675],'b':[-0.12006616 , 1.18527028, -3.54715528,  2.72209057],'c':[-0.06447898,  0.87229084, -2.47581575,  2.03771726],'d':[-0.01303147,  0.13568383, -0.38909038,  0.32880047]}}


prms={'Uu':{'a':[4.3761815131090005,-11.761147930663759,11.094653118471262]},
'Vu':{'a':[4.036769646958779,-8.6952237574292,6.665209781616562]},
'Wu':{'a':[  0.63163208 ,  3.88577255 ,-10.12068558  , 7.72460201]},
'Tu':{'a':[0.27814402332658494,-1.3300825904354217,1.8360360499380333]},
'H2Ou':{'a':[7.383821157007216,-16.816367865048143,36.85379157148182,-22.552696633480064]},
'CO2u':{'a':[11.627039011171679,-55.75662476948497,132.1688475947626,-91.53182129882329]},
'Us':{'a':[1.4717151872327583,0.012072368555451676,1.899822307024067],'b':[0.19323057864318993,-0.30596741646111014,0.20047175184453506]},
'Vs':{'a':[2.3403371987864494,-2.3209024377836824,3.327691452120783],'b':[-0.015725591952585882,0.5966752364174301,-1.1158220609457767,0.6679697765221119]},
'Ws':{'a':[0.8614591430983672,0.37304223945564274,2.419204148353563],'b':[0.0785398588770194,0.12596738442976788,-0.3818551619370662,0.2928223953421529]},
'Ts':{'a':[0.5508850263994217,-0.6010721359441746],'b':[-0.045603526614089944,0.6295884954506813,-1.4204154235691635,0.7646404325025351],'c':[0.03559294900969055,-0.5314751867766699,1.228798456650419,-0.6927622685957189],'d':[-0.015059246695466913,-0.07956296754856743,0.07803533743477396]},
'H2Os':{'a':[0.6847397918306842,-0.5357826052801081],'b':[-0.07008794959354223,0.7296454546728901,-2.248936751108793,1.7535328541026547],'c':[-0.02650127134959892,0.33528955869577337,-1.2926670886115836,1.265495431414254],'d':[-0.00558767651372621,0.05144230881632421,-0.22070545719653586,0.23276548747625145]},
'CO2s':{'a':[0.6131591177739361,-0.5179715251273849],'b':[-0.12006121500021293,1.1852189407405058,-3.5470108622360246,2.7219768701090667],'c':[-0.0644922347355642,0.8724211109027918,-2.476185445675213,2.03801562775291],'d':[-0.013033310609906788,0.13570473638736016,-0.38915302565743964,0.32885620239891916]}}

prms={'Uu':{'a':[4.3761815131090005,-11.761147930663759,11.094653118471262]},
'Vu':{'a':[4.036769646958779,-8.6952237574292,6.665209781616562]},
'Wu':{'a':[  0.63163208 ,  3.88577255 ,-10.12068558  , 7.72460201]},
'Tu':{'a':[0.27814402332658494,-1.3300825904354217,1.8360360499380333]},
'H2Ou':{'a':[7.383821157007216,-16.816367865048143,36.85379157148182,-22.552696633480064]},
'CO2u':{'a':[11.627039011171679,-55.75662476948497,132.1688475947626,-91.53182129882329]},
'Us':{'a':[1.4717151872327583,0.012072368555451676,1.899822307024067],'b':[0.19323057864318993,-0.30596741646111014,0.20047175184453506]},
'Vs':{'a':[2.3403371987864494,-2.3209024377836824,3.327691452120783],'b':[-0.015725591952585882,0.5966752364174301,-1.1158220609457767,0.6679697765221119]},
'Ws':{'a':[0.8614591430983672,0.37304223945564274,2.419204148353563],'b':[0.0785398588770194,0.12596738442976788,-0.3818551619370662,0.2928223953421529]},
'Ts':{'a':[0.0016086400426415747,-0.009046112626421346,0.01674919857041958],'b':[3.1069569610314867,-2.721337687613489]},
'H2Os':{'a':[3.5838553759472704,-25.592585974821585,63.95066854368746,-39.75572561544258],'b':[0.42382342649041355,24.70142140535896,-68.94899093260442,41.594606950484305]},
'CO2s':{'a':[0.8845657789177256,-0.5835075212388595,2.237996599356271],'b':[3.1701608838942703,-4.123288804470784]}}

# tsw_v4
prms={'Uu':{'a':[4.123621495800058,-9.019103428856788,7.031233766838203]},
'Vu':{'a':[4.287424385993223,-11.432206917682109,10.714434337994307]},
'Wu':{'a':[0.8322246683633081,2.2982094340935624,-5.9245962466602675,4.359932190370337]},
'Us':{'a':[2.48999325402708,-2.9309894919097794,3.9665235021601855],'b':[-0.014091947587046363,0.4413024489123778,-0.4734026436270551]},
'Vs':{'a':[1.3330870632847518,0.5841790794590472,1.3070551206541443],'b':[0.22831158178720407,-0.48384488258397285,0.48954414345477587,-0.14416529116966556]},
'Ws':{'a':[0.8568248498732863,0.3850311195961679,2.398970306113473],'b':[0.08589790229105511,0.09024112195753205,-0.3132154689043823,0.24661181960543574]}}

# tsw_w
prms={'Uu':{'a':[3.3615175860280218,-6.925700629790642,6.232958203549107]},
'Vu':{'a':[3.0557922880978188,-6.920145232464449,7.278468425023672]},
'Wu':{'a':[0.9787894467148208,1.4560953767123934,-5.146119437286295,4.576692940662369]},
'Us':{'a':[2.313825066597639,-2.5689807078493745,1.9183079661659839],'b':[0.005799653441426352,0.5373072278844293,-0.6884263128616442]},
'Vs':{'a':[1.828731530567526,-1.1835102171131944,0.9066699531926097],'b':[0.0880768048774914,0.6628429400053683,-2.386858617199514,2.163448436590539]},
'Ws':{'a':[1.1701429855606926,0.596408748744598,-0.6677632827805118],'b':[0.13975461258799823,0.5982347407850199,-2.367596062889779,2.3076983900316668]}}

# tsw_w (but log)
prms={'Uu':{'a':[1.0610229608801973,-1.2757524443591375,0.49143888423560594]},
'Vu':{'a':[1.279919385369563,-1.053292182683859,-1.7795853192407012,-2.167444842040447]},
'Wu':{'a':[0.9930741960252398,0.21810289143827763,1.1612334093298597,0.8863814525535234]},
'Us':{'a':[1.1535322771722971,-1.2713070345001816,-0.3427488963602023],'b':[0.07216762468514018,-0.14404086512357264,-0.17237626708802795]},
'Vs':{'a':[1.2980322066263459,-0.5857409086926993,-0.16275214215115155],'b':[0.06842778845656437,-0.06686439770911787,0.20237632829121496,0.21044999635345177]},
'Ws':{'a':[1.1701429855606926,0.596408748744598,-0.6677632827805118],'b':[0.13975461258799823,0.5982347407850199,-2.367596062889779,2.3076983900316668]}}

letters=['a','b','c','d']


def get_params(v,ani_in):
    outlist=[]
    Np=len(prms[v].keys())
    for i in range(Np):
        if lg[v][i]:
            ani=np.log10(ani_in)
        else:
            ani=ani_in
        out=0
        for j in range(len(prms[v][letters[i]])):
            out=out+prms[v][letters[i]][j]*ani**j
        outlist.append(out)
    return outlist


def binplot1d(xx,yy,ani,stb,xbins=-zLbins,anibins=anibins):
    if stb:
        xbins=-xbins[-1:0:-1]
    xplot=(xbins[0:-1]+xbins[1:])/2
    yplot=np.zeros((len(anibins)-1,len(xplot)))
    count=np.zeros((len(anibins)-1,len(xplot)))
    aniplot=(anibins[0:-1]+anibins[1:])/2
    for i in range(len(anibins)-1):
        #print('ANI: '+str(anibins[i]))
        for j in range(len(xbins)-1):
            pp=get_params(var,aniplot)
            Np=len(pp)
            if Np==1:
                yplot[i,j]=fxns_[var](xplot[j],pp[0][i])
            elif Np==2:
                yplot[i,j]=fxns_[var](xplot[j],pp[0][i],pp[1][i])
            elif Np==3:
                yplot[i,j]=fxns_[var](xplot[j],pp[0][i],pp[1][i],pp[2][i])
            elif Np==4:
                yplot[i,j]=fxns_[var](xplot[j],pp[0][i],pp[1][i],pp[2][i],pp[3][i])
            #print('  ZL: '+str(xbins[j]))
    return xplot,yplot,aniplot



#####################

# unstable  -> species -> [p_phi,p_zL,phi,zL,ani,SS,SSlo,SShi,SS_s,SSlo_s,SShi_s]
# stable    -> species -> [...]
# aniso_all -> sites   -> [yb,xb] -> [lc,median,pct90,pct10,std]
# aniso_stb -> sites   -> [yb,xb] -> [full,hi,lo] -> [lc,median,pct90,pct10,std]
# aniso_ust -> sites   -> [yb,xb] -> [full,hi,lo] -> [lc,median,pct90,pct10,std]

#fxns_={'Uu':U_ust,'Vu':U_ust,'Wu':W_ust,'Tu':T_ust,'H2Ou':C_ust,'CO2u':C_ust,\
#       'Us':U_stb,'Vs':U_stb,'Ws':W_stb,'Ts':T_stb,'H2Os':C_stb,'CO2s':C_stb2}

fxns_={'Uu':U_ust,'Vu':U_ust,'Wu':W_ust,\
       'Us':U_stb,'Vs':U_stb,'Ws':W_stb}


d_unst={}
d_stbl={}

for var in fxns_.keys():
    print(var,flush=True)
    # Identify the filename to load in the data
    fname='/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/NEON_TW_'
    if 'u' in var:
        d_=d_unst
        fname=fname+'U_'
        zLbins=-np.logspace(-4,2,40)[-1:0:-1]
        stb=False
    else:
        d_=d_stbl
        fname=fname+'S_'
        zLbins=np.logspace(-4,2,40)
        stb=True
    if ('U' in var) or ('V' in var) or ('T' in var) or ('W' in var):
        fname=fname+'UVWT.h5'
    elif ('H2O' in var):
        fname=fname+'H2O.h5'
    else:
        fname=fname+'CO2.h5'
    fp=h5py.File(fname,'r')
    fpsite=fp['SITE'][:]

    d_[var[0:-1]]={}


    phi_,ani_,zL_ = get_phi(fp,var)

    mmm=~np.isnan(phi_)
    phi_=phi_[mmm]
    ani_=ani_[mmm]
    zL_=zL_[mmm]
    fpsite=fpsite[mmm]

    #def add_species(d_,sp,phi,zL,ani,phi_new,phi_old,m,fpsite,stb=False):
    Np=len(inspect.signature(fxns_[var]).parameters)-1
    pp=get_params(var,ani_)

    if Np==1:
        phi_new=fxns_[var](zL_,pp[0])
    elif Np==2:
        phi_new=fxns_[var](zL_,pp[0],pp[1])
    elif Np==3:
        phi_new=fxns_[var](zL_,pp[0],pp[1],pp[2])
    elif Np==4:
        phi_new=fxns_[var](zL_,pp[0],pp[1],pp[2],pp[3])

    x_,y_,z_=binplot1d(zL_,phi_,ani_,stb)

    d_[var[0:-1]]['p_phi']=y_
    d_[var[0:-1]]['p_zL']=x_
    d_[var[0:-1]]['p_ani']=z_
    d_[var[0:-1]]=add_species(d_,var[0:-1],phi_,zL_,ani_,phi_new,old(var,zL_),fpsite,stb)


pickle.dump(d_unst,open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_unst_tw_w.p','wb'))
pickle.dump(d_stbl,open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_stbl_tw_w.p','wb'))

#pickle.dump(d_unst,open('/home/tsw35/soteria/neon_advanced/data/d_unst_tw_v2.p','wb'))
#pickle.dump(d_stbl,open('/home/tsw35/soteria/neon_advanced/data/d_stbl_tw_v2.p','wb'))

