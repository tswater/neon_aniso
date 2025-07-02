# curve fitting functions






##############################################################################
##############################################################################
##############################################################################
##########                                                          ##########
##########                  CURVE FIT CLASS                         ##########
##########                                                          ##########
##############################################################################
##############################################################################
class Cfit:

    def __init__(self,name,var,stab,ftype,base):
        #### USER DEFINED
        self.name=name # name of fit
        self.var=var # var being fitted;
        self.ftype=ftype # type of fit 'full' or '2_stage'
        self.stab=stab # stability boolean
        self.base=base # base function

        #### INITIALIZED
        self.p0=None
        self.params=None
        self.const={'a':None,'b':None,'c':None,'d':None,'e':None}
        self.bnds=None
        self.loss=None
        self.in2=None

    def cfit(self,zL,ani,phi,p0=self.p0,params=self.params,const=self.const,bnds=self.bnds,loss=self.loss,ftype=self.ftype,in2=self.in2):
        # remove any statistics
        self.stats=None

        if p0!=self.p0:
            self.p0=p0
        if params!=self.params:
            self.params=params
        if const!=self.const:
            self.const=const
        if bnds!=self.bnds:
            self.bnds=bnds
        if loss!=self.loss:
            self.p0=p0
        if ftype!=self.ftype:
            self.ftype=ftype

        # Generate function
        fxn = fxngen(self.base,ac_=const['a'],bc_=const['b'],cc_=const['c'],dc_=const['d'],ec_=const['e'])

        if ftype=='2_stage':
            if (in2 in [None]):
                # do stage 1

            # do stage 2

        if ftype=='full':
            popt,pcov=optimize.curve_fit(fxn,[zL,ani],phi,p0,bnds,loss=loss)

        self.popt=popt
        self.pcov=pcov
        return popt,pcov
















def cfit_p1(fxn,var,bnd,p0,nbin,fpu,fps,exclude=0,loss='cauchy'):
    '''
    fxn : functional form to fit
    var : variable and stability in format [variable][stability] i.e. TU
    zL  : stability
    ani : anisotropy yb
    fps : file with data stable
    fpu : file with data unstable
    bnd : boundaries
    nbin: number of bins
    p0  : initial conditions
    '''

    # computed variables
    Np = len(p0)
    if var[1]=='U':
        fp=fpu
    else:
        fp=fps
    zL=fp['zzd'][:]/fp['L_MOST'][:]
    ani=fp['ANI_YB'][:]
    phi,_ = get_phi(fp,zL,var)
    abins,mm = binit(ani,n=nbin)
    ani_=ani[mm]
    zL_=zL[mm]
    phi_=phi[mm]
    params=np.ones((len(abins)-1,Np))*float('nan')

    match var:
        case 'TU':
            flx_=fp['H'][mm]
        case 'TS':
            flx_=fp['H'][mm]
        case 'QU':
            flx_=fp['LE'][mm]
        case 'QS':
            flx_=fp['LE'][mm]
        case 'CU':
            flx_=fp['CO2FX'][mm]
        case 'CS':
            flx_=fp['CO2FX'][mm]*10**(6)


    for i in range(len(abins)-1):
        m=(ani_>abins[i])&(ani_<abins[i+1])&(~np.isnan(phi_))
        print(i)
        if exclude>0:
            m=m&(np.abs(flx_)>exclude)
        try:
            params[i,:],pcov=optimize.curve_fit(fxn,zL_[m],phi_[m],p0,bounds=bnd, loss=loss)
        except Exception as e:
            print(fxn)

    in2=(fxn,var,params,abins,Np)
    return in2

def cfit_p2(in2,deg,typ,plot1=True):
    '''
    in2    : contains fxn, var, params
    typ    : list of type of curve fitting. Options currently: polynomial, logarithmic. Consider: exponential, hill function, power
    deg    : list of degrees of fit for each parameter [deg_a,deg_b,deg_c]
    plot1  : boolean, whether to produce plots
    '''
    fxn=in2[0]
    var=in2[1]
    param=in2[2]
    abins=in2[3]
    anilvls=(abins[1:]+abins[0:-1])/2
    m=~np.isnan(param[:,0])
    p_prime=[]
    for i in range(param.shape[1]):
        x=anilvls[m]
        match typ[i]:
            case 'log':
                x=np.log10(anilvls[m])
            case 'lin':
                x=anilvls[m]
            case 'exp':
                x=np.exp(anilvls[m])
        y=param[:,i][m]
        c=P.polyfit(x,y,deg[i])
        p_prime.append(c)

        if plot1:
            plt.figure(figsize=(3,6))
            plt.subplot(in2[4],1,i+1)
            plt.scatter(x,param[:,i][m])
            y2=c[0]+c[1]*x
            if deg[i]>1:
                y2=y2+c[2]*x**2
            if deg[i]>2:
                y2=y2+c[3]*x**3
            if deg[i]>3:
                y2=y2+c[4]*x**4
            plt.plot(x,y2,'k--')
            plt.title(var+' parameter '+str(i))

    in3=(fxn,var,in2[4],typ,deg,p_prime)
    # report contents: fxn,var,Np,type,deg,p_prime
    return in3

###############################################################################
################# EVALUATE a CURVE FIT ########################################
# Evaluate a curvefit by generating a comprehensive report
def eval_cfit(fxn,var,fpu,fps,pnames,pvals,pcov,plotit=True,plotonly=False):
    if var[1]=='U':
        fp=fpu
    else:
        fp=fps
    zL=fp['zzd'][:]/fp['L_MOST'][:]
    ani=fp['ANI_YB'][:]
    phi,phio = get_phi(fp,zL,var)
    pdict={}
    for i in range(len(pnames)):
        pdict[pnames[i]]=pvals[i]
    phin=fxn([zL,ani],**pdict)

    if plotit:
        m=np.zeros((len(phin),)).astype(bool)
        m[0:10000]=True
        np.random.shuffle(m)
        plt.figure()
        if var[1]=='U':
            plt.semilogx(-zL[m],phi[m],'o',markersize=1,color='grey',alpha=.3)
            plt.semilogx(-zL[m],phio[m],'o',markersize=1,color='black')
            zLi=-np.logspace(-3,2)
            phi1=fxn([zLi,np.ones((50,))*.1],**pdict)
            phi3=fxn([zLi,np.ones((50,))*.3],**pdict)
            phi5=fxn([zLi,np.ones((50,))*.5],**pdict)
            plt.semilogx(-zLi,phi1,color='red')
            plt.semilogx(-zLi,phi3,color='tan')
            plt.semilogx(-zLi,phi5,color='blue')
            plt.xlim(10**(-3),10**(1.5))
            plt.ylim(.1,20)
            plt.gca().invert_xaxis()
        else:
            plt.loglog(zL[m],phi[m],'o',markersize=1,color='grey',alpha=.3)
            plt.loglog(zL[m],phio[m],'o',markersize=1,color='black')
            zLi=np.logspace(-3,2)
            phi1=fxn([zLi,np.ones((50,))*.1],**pdict)
            phi3=fxn([zLi,np.ones((50,))*.3],**pdict)
            phi5=fxn([zLi,np.ones((50,))*.5],**pdict)
            phi7=fxn([zLi,np.ones((50,))*.7],**pdict)
            plt.loglog(zLi,phi1,color='red')
            plt.loglog(zLi,phi3,color='orange')
            plt.loglog(zLi,phi5,color='tan')
            plt.loglog(zLi,phi7,color='blue')
            plt.xlim(10**(-3),10**(1.5))
            plt.ylim(.5,20)
        if plotonly:
            return None

    mlo=np.abs(zL)<.1
    mhi=np.abs(zL)>.1

    mdo=np.nanmedian(np.abs(phi-phio))
    mdn=np.nanmedian(np.abs(phi-phin))
    mdol=np.nanmedian(np.abs(phi[mlo]-phio[mlo]))
    mdnl=np.nanmedian(np.abs(phi[mlo]-phin[mlo]))
    mdoh=np.nanmedian(np.abs(phi[mhi]-phio[mhi]))
    mdnh=np.nanmedian(np.abs(phi[mhi]-phin[mhi]))

    ss=1-mdn/mdo
    sslo=1-mdnl/mdol
    sshi=1-mdnh/mdoh

    sslo_s=[]
    ss_s=[]
    sshi_s=[]

    fpsites=fp['SITE'][:]
    sites=np.unique(fpsites)
    sites.sort()
    for site in sites:
        ms=fpsites==site
        mlo=(np.abs(zL)<.1)&ms
        mhi=(np.abs(zL)>.1)&ms
        mdo=np.nanmedian(np.abs(phi[ms]-phio[ms]))
        mdn=np.nanmedian(np.abs(phi[ms]-phin[ms]))
        mdol=np.nanmedian(np.abs(phi[mlo]-phio[mlo]))
        mdnl=np.nanmedian(np.abs(phi[mlo]-phin[mlo]))
        mdoh=np.nanmedian(np.abs(phi[mhi]-phio[mhi]))
        mdnh=np.nanmedian(np.abs(phi[mhi]-phin[mhi]))
        ss_s.append(1-mdn/mdo)
        sslo_s.append(1-mdnl/mdol)
        sshi_s.append(1-mdnh/mdoh)

    ss_s=np.array(ss_s)
    sslo_s=np.array(sslo_s)
    sshi_s=np.array(sshi_s)

    sgss=np.nanstd(ss_s)
    sgsslo=np.nanstd(sslo_s)
    sgsshi=np.nanstd(sshi_s)
    mss=np.nanmedian(ss_s)
    msslo=np.nanmedian(sslo_s)
    msshi=np.nanmedian(sshi_s)

    Nworse=np.sum(ss_s<0)

    time=datetime.now()

    report={}
    report['time']=time
    report['fxn']=fxn
    report['var']=var
    report['type']=''
    report['deg']=''
    report['pcov']=np.sqrt(np.diag(pcov))
    report['p_prime']=pvals
    report['SS']=ss
    report['SSlo']=sslo
    report['SShi']=sshi
    report['sigSS']=sgss
    report['sigSSlo']=sgsslo
    report['sigSShi']=sgsshi
    report['medSS']=mss
    report['medSSlo']=msslo
    report['medSShi']=msshi
    report['N_worse']=Nworse
    report['site_SS']=ss_s
    report['sites']=sites

    # SS,SSlo,SSmd,SShi,sigSS,mSS,mSSlo,mSSmd,mSShi,Nworse

    # fxn,var,type,deg,param_values,SS,SSlo,SSmd,SShi,sigSS,mSS,mSSlo,mSSmd,mSShi,Nworse
    return report


#############################################################################
##################### COMPARE CURVE FIT RESULTS #############################
# Compares the results of different curve fits
def compare_fits(reports):
    ''' turns a list of reports into useful output '''
    names=['idx|ID-----','-SCORE-','-CMPLX-','--SS---','-medSS-','-sigSS-','-mSSlo-','-mSShi-','Nworse-']
    vari=[]
    for rp in reports:
        vari.append(rp['var'])
    vari=np.array(vari)
    for v in np.unique(vari):
        N=np.sum(vari==v)
        full=np.zeros((N,len(names)))
        for i in range(N):
            idx=np.where(vari==v)[0][i]
            rp=reports[idx]
            full[i,0]=idx
            a=len(rp['p_prime'])
            full[i,2]=a
            full[i,3]=rp['SS']
            full[i,4]=rp['medSS']
            full[i,5]=rp['sigSS']
            full[i,6]=rp['medSSlo']
            full[i,7]=rp['medSShi']
            full[i,8]=rp['N_worse']

            full[i,1]=(rp['SS']/.2*5+rp['medSS']/.2*15-rp['sigSS']/.58*5-rp['N_worse']/4\
                    -10*(np.abs(rp['medSSlo']-rp['medSS'])+np.abs(rp['medSShi']-rp['medSS'])))-(a)/2

        print('---------------------------------'+str(v)+'-------------------------------------')
        for n in names:
            print(n,end='|')
        print()
        for i in range(N):
            idx=np.where(vari==v)[0][i]
            rp=reports[idx]
            for n in range(len(names)):
                if n==0:
                    ts='{0:3}'.format(int(full[i,n]))+str('|')
                    print(ts,end='')
                    print('{0:7}'.format(rp['fxn'].__name__)+str('|'),end='')
                elif full[i,n]<0:
                    ts='{0: 6}'.format(full[i,n])
                    ts=ts[0:7]+str('|')
                    print(ts,end='')
                else:
                    ts=' {0: 6}'.format(full[i,n])
                    ts=ts[0:7]+str('|')
                    print(ts,end='')
            print()
        print()

########################################################################
######################### BASEFUNCTIONS ################################
# Basefunctions for function generator

#### CORE BASE
# the basic base function from theory
def cbase(zL,a,b,c,d,e):
    return a*(b+c*zL)**(d)+e

#########################################################################
######################## FUNCTION GENERATOR #############################
# generate a function for curve fitting
def fxngen(plist,base,ac_=None,bc_=None,cc_=None,dc_=None,ec_=None):
    lpms=4
    n=len(plist)
    a=[0,0,0,0]
    b=[0,0,0,0]
    c=[0,0,0,0]
    d=[0,0,0,0]
    e=[0,0,0,0]
    for i in range(n):
        param=plist[i]
        if 'a' in param:
            a[int(param[1])]=1
        elif 'b' in param:
            b[int(param[1])]=1
        elif 'c' in param:
            c[int(param[1])]=1
        elif 'd' in param:
            d[int(param[1])]=1
        elif 'e' in param:
            e[int(param[1])]=1
    def fxn(inp,*prms):
        zL=inp[0]
        ani=inp[1]
        if len(prms) !=n:
            return np.ones(zL.shape)*float('nan')
        pm=0
        if ac_ in [None]:
            aa=np.zeros((len(ani),))
            for i in range(lpms):
                if a[i]==1:
                    aa=aa+prms[pm]*ani**i
                    pm=pm+1
        else:
            aa=ac_
        if bc_ in [None]:
            bb=np.zeros((len(ani),))
            for i in range(lpms):
                if b[i]==1:
                    bb=bb+prms[pm]*ani**i
                    pm=pm+1
        else:
            bb=bc_
        if cc_ in [None]:
            cc=np.zeros((len(ani),))
            for i in range(lpms):
                if c[i]==1:
                    cc=cc+prms[pm]*ani**i
                    pm=pm+1
        else:
            cc=cc_
        if dc_ in [None]:
            dd=np.zeros((len(ani),))
            for i in range(lpms):
                if d[i]==1:
                    dd=dd+prms[pm]*ani**i
                    pm=pm+1
        else:
            dd=dc_
        if ec_ in [None]:
            ee=np.zeros((len(ani),))
            for i in range(lpms):
                if e[i]==1:
                    ee=ee+prms[pm]*ani**i
                    pm=pm+1
        else:
            ee=ec_
        return base(zL,aa,bb,cc,dd,ee)
    return fxn







