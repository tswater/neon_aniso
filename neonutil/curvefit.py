# curve fitting functions
import numpy as np
import h5py
import ast
import datetime
from numpy.polynomial import polynomial as P
from scipy import optimize
try:
    from nutil import SITES,get_phi,get_phio
except:
    from neonutil.nutil import SITES,get_phi,get_phio



try:
    import matplotlib.pyplot as plt
except:
    pass


def _nsv(d):
    if d is None:
        return 'NA'
    try:
        if d in [None,[],'',float('nan')]:
            return 'NA'
        else:
            return d
    except Exception:
        return d


########################################################################
######################### BASEFUNCTIONS ################################
# Basefunctions for function generator

#### CORE BASE
# the basic base function from theory

def cbase(zL,a,b,c,d,e):
    return a*(b+c*zL)**(d)+e




#### LIST OF BASES
bases={'cbase':cbase}

##############################################################################
##############################################################################
##############################################################################
##########                                                          ##########
##########                  CURVE FIT CLASS                         ##########
##########                                                          ##########
##############################################################################
##############################################################################
class Cfit:

    def __init__(self,name,var,stab,ftype,base,p0=None,bnds=None,loss=None,\
            const={'a':None,'b':None,'c':None,'d':None,'e':None},params=None,\
            in2=None,deg=None,typ=None):
        #### USER DEFINED
        self.name=name # name of fit
        self.var=var # var being fitted;
        self.ftype=ftype # type of fit 'full' or '2_stage'
        self.stab=stab # stability boolean
        self.base=base # base function

        #### INITIALIZED
        self.p0=p0
        self.const=const
        self.bnds=bnds
        self.loss=loss
        # for 'full' only
        self.params=params
        # for '2_stage' only
        self.in2=in2
        self.deg=deg
        self.typ=typ

        self.stats=None
        self.popt=None
        self.pcov=None

    def _add_stats(self,popt=None,pcov=None,stats=None):
        self.popt=popt
        self.pcov=pcov
        self.stats=stats

    def cfit(self,zL,ani,phi,plot=True,nbin=50,p0=None,params=None,\
            const=None,bnds=None,loss=None,ftype=None,\
            deg=None,typ=None):

        if p0==None:
            p0=self.p0
        if params==None:
            params=self.params
        if const==None:
            const=self.const
        if bnds==None:
            bnds=self.bnds
        if loss==None:
            loss=self.loss
        if ftype==None:
            ftype=self.ftype
        if deg==None:
            deg=self.deg
        if typ==None:
            typ=self.typ

        # remove any statistics
        self.stats=None
        self.in2=None

        # Generate function
        if ftype=='2_stage':
            if (self.in2 in [None]):
                # do stage 1
                plist=[]
                for k in const.keys():
                    if const[k]==None:
                        plist.append(k+'0')
                fxn = fxngen(plist,bases[self.base],ac_=self.const['a'],\
                        bc_=self.const['b'],cc_=self.const['c'],\
                        dc_=self.const['d'],ec_=self.const['e'])
                self.in2=cfit_p1(fxn,zL,ani,phi,p0=p0,bnd=self.bnds,nbin=nbin,loss=loss)

            # do stage 2
            in3=cfit_p2(self.in2,self.deg,self.typ,plot1=True)
            p_prime=in3[4]
            popt=self._convert_out(p_prime)
            pcov=np.ones((len(popt),))*float('nan')
            self.params=self._convert_in(self.const,self.deg)

        if ftype=='full':
            fxn = fxngen(self.params,bases[self.base],ac_=const['a'],bc_=const['b'],cc_=const['c'],dc_=const['d'],ec_=const['e'])
            popt,pcov=optimize.curve_fit(fxn,[zL,ani],phi,p0=p0,bounds=bnds,loss=loss)

        self.popt=popt
        self.pcov=pcov
        return popt,pcov

    def _convert_out(self,p_prime):
        # convert p_prime to popt
        popt=[]
        for i in range(len(p_prime)):
            for j in range(len(p_prime[i])):
                popt.append(p_prime[i][j])
        return popt

    def _convert_in(self,const,deg):
        # convert deg to params
        pnames=[]
        for k in const.keys():
            if const[k]==None:
                pnames.append(k)
        params=[]
        for i in range(len(pnames)):
            for j in range(deg[i]+1):
                params.append(pnames[i]+str(j))
        return params

    def eval_cfit(self,zL,ani,phi,fpsites,const=None,plotit=True,plotonly=False):

        params=self.params
        pvals=self.popt
        const=self.const
        var=self.var
        pcov=self.pcov

        print(params)
        print(pvals)
        print(const)
        print(var)

        fxn = fxngen(self.params,bases[self.base],ac_=const['a'],bc_=const['b'],cc_=const['c'],dc_=const['d'],ec_=const['e'])
        phio = get_phio(var,self.stab,zL=zL)
        phin=fxn([zL,ani],*pvals)

        if plotit:
            m=np.zeros((len(phin),)).astype(bool)
            m[0:10000]=True
            np.random.shuffle(m)
            plt.figure()
            if not self.stab:
                plt.semilogx(-zL[m],phi[m],'o',markersize=1,color='grey',alpha=.3)
                plt.semilogx(-zL[m],phio[m],'o',markersize=1,color='black')
                zLi=-np.logspace(-3,2)
                phi1=fxn([zLi,np.ones((50,))*.1],*pvals)
                phi3=fxn([zLi,np.ones((50,))*.3],*pvals)
                phi5=fxn([zLi,np.ones((50,))*.5],*pvals)
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
                phi1=fxn([zLi,np.ones((50,))*.1],*pvals)
                phi3=fxn([zLi,np.ones((50,))*.3],*pvals)
                phi5=fxn([zLi,np.ones((50,))*.5],*pvals)
                phi7=fxn([zLi,np.ones((50,))*.7],*pvals)
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

        time=datetime.datetime.now()

        report={}
        report['time']=time
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
        report['SS_site']=ss_s
        report['SSlo_site']=sslo_s
        report['SShi_site']=sshi_s

        self.stats=report

    def save_fit(self,fg):
        # save to L2 file where fg is the group not the file
        if 'curve_fits' not in fg.keys():
            fg.create_group('curve_fits')
        stbstr=''
        if self.stab:
            stbstr='stable'
        else:
            stbstr='unstable'
        try:
            f_=fg['curve_fits'].create_group(self.var+'::'+stbstr+'::'+self.name)
        except ValueError:
            f_=fg['curve_fits/'+self.var+'::'+stbstr+'::'+self.name]
        f_.attrs['var']=self.var
        f_.attrs['ftype']=self.ftype
        f_.attrs['stab']=self.stab
        f_.attrs['base']=self.base
        f_.attrs['p0']=list(_nsv(self.p0))
        f_.attrs['loss']=_nsv(self.loss)
        f_.attrs['bnds']=_nsv(self.bnds)
        for k in self.const.keys():
            if self.const[k]==None:
                pass
            else:
                f_.attrs[k]=self.const[k]

        tmp={}
        for k in self.params:
            if k[0] not in tmp.keys():
                tmp[k[0]]=[]
            if int(k[1]) not in tmp[k[0]]:
                tmp[k[0]].append(int(k[1]))
        for k in tmp.keys():
            f_.attrs[k]=tmp[k]

        if self.ftype=='2_stage':
            f_.attrs['typ']=str(_nsv(self.typ))
            f_.attrs['deg']=_nsv(self.deg)

        f_.attrs['popt']=_nsv(self.popt)
        f_.attrs['pcov']=_nsv(self.pcov)

        if self.stats not in [None,{},[],'NA']:
            try:
                fs=f_.create_group('stats')
            except ValueError:
                fs=f_['stats']
            try:
                fs.create_dataset('SS_site',data=self.stats['SS_site'][:])
            except ValueError:
                del fs['SS_site']
                fs.create_dataset('SS_site',data=self.stats['SS_site'][:])

            try:
                fs.create_dataset('SSlo_site',data=self.stats['SSlo_site'][:])
            except ValueError:
                del fs['SSlo_site']
                fs.create_dataset('SSlo_site',data=self.stats['SSlo_site'][:])

            try:
                fs.create_dataset('SShi_site',data=self.stats['SShi_site'][:])
            except ValueError:
                del fs['SShi_site']
                fs.create_dataset('SShi_site',data=self.stats['SShi_site'][:])


            for k in self.stats.keys():
                if k in ['SS_site','SSlo_site','SShi_site']:
                    continue
                else:
                    try:
                        fs.attrs[k]=_nsv(self.stats[k])
                    except TypeError:
                        fs.attrs[k]=str(self.stats[k])

    def get_phin(self,zL,ani):
        fxn = fxngen(self.params,bases[self.base],ac_=self.const['a'],\
                bc_=self.const['b'],cc_=self.const['c'],dc_=self.const['d'],\
                ec_=self.const['e'])
        return fxn([zL,ani],*self.popt)

####################### END OF CLASS Cfit ####################################
##############################################################################
##############################################################################
##############################################################################

def load_fit(fp,casek,name,var=None,stab=None):
    # load fit from an L2
    if '::' in name:
        nm=name
        name=nm.split('::')[2]
    elif (var==None) or (stab==None):
        raise ValueError('if name is not full name of curve fit, var and '+\
                'stability must be specified')
    else:
        if type(stab)==bool:
            if stab:
                stbstr='stable'
            else:
                stbstr='unstable'
        if type(stab)==str:
            stbstr=stab
        nm=var+'::'+stbstr+'::'+name
    fa=fp[casek]['curve_fits'][nm].attrs
    fs=fp[casek]['curve_fits'][nm]['stats']

    const={}
    params=[]
    for k in ['a','b','c','d','e']:
        if k=='a':
            continue
        if type(fa[k])==np.ndarray:
            for j in fa[k]:
                params.append(k+str(j))
            const[k]=None
        else:
            const[k]=fa[k]

    deg=None
    typ=None
    try:
        deg=fa['deg']
    except Exception:
        pass
    try:
        typ=ast.literal_eval(fa['typ'])
    except Exception:
        pass
    cfit=Cfit(name,fa['var'],fa['stab'],fa['ftype'],fa['base'],\
            p0=fa['p0'],bnds=fa['bnds'],loss=fa['loss'],\
            const=const,params=params,deg=deg,\
            typ=typ)

    # load stats
    stats={}
    try:
        stats['SS_site']=fs['SS_site'][:]
    except KeyError:
        pass

    try:
        stats['SSlo_site']=fs['SSlo_site'][:]
    except KeyError:
        pass

    try:
        stats['SShi_site']=fs['SShi_site'][:]
    except KeyError:
        pass

    for k in fs.attrs.keys():
        stats[k]=fs.attrs[k]
    cfit._add_stats(popt=fa['popt'],pcov=fa['pcov'],stats=stats)

    return cfit


def cfit_p1(fxn,zL,ani,phi,p0,bnd,nbin=50,loss='cauchy'):
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
    Np=len(p0)
    abins,mm = binit(ani,n=nbin)
    ani_=ani[mm]
    zL_=zL[mm]
    phi_=phi[mm]
    popt=np.ones((len(abins)-1,Np))*float('nan')

    for i in range(len(abins)-1):
        m=(ani_>abins[i])&(ani_<abins[i+1])&(~np.isnan(phi_))
        try:
            popt[i,:],pcov=optimize.curve_fit(fxn,[zL_[m],ani_[m]],phi_[m],p0,bounds=bnd, loss=loss)
        except Exception as e:
            print(e)

    in2=(fxn,popt,abins,Np)
    return in2

def cfit_p2(in2,deg,typ,plot1=True):
    '''
    in2    : contains fxn, var, params
    typ    : list of type of curve fitting. Options currently: polynomial, logarithmic. Consider: exponential, hill function, power
    deg    : list of degrees of fit for each parameter [deg_a,deg_b,deg_c]
    plot1  : boolean, whether to produce plots
    '''
    fxn=in2[0]
    popt=in2[1]
    abins=in2[2]
    anilvls=(abins[1:]+abins[0:-1])/2
    m=~np.isnan(popt[:,0])
    p_prime=[]
    for i in range(popt.shape[1]):
        x=anilvls[m]
        match typ[i]:
            case 'log':
                x=np.log10(anilvls[m])
            case 'lin':
                x=anilvls[m]
            case 'exp':
                x=np.exp(anilvls[m])
        y=popt[:,i][m]
        c=P.polyfit(x,y,deg[i])
        p_prime.append(c)

        if plot1:
            plt.figure(figsize=(3,6))
            plt.subplot(in2[3],1,i+1)
            plt.scatter(x,popt[:,i][m])
            y2=c[0]+c[1]*x
            if deg[i]>1:
                y2=y2+c[2]*x**2
            if deg[i]>2:
                y2=y2+c[3]*x**3
            if deg[i]>3:
                y2=y2+c[4]*x**4
            plt.plot(x,y2,'k--')
            plt.title('parameter '+str(i))

    in3=(fxn,in2[3],typ,deg,p_prime)
    # report contents: fxn,var,Np,type,deg,p_prime
    return in3


def binit(ani,binsize=float('nan'),n=100,vmx_a=.7,vmn_a=.15):
    mmm=(ani>=vmn_a)&(ani<=vmx_a)
    N=np.sum(mmm)
    if np.isnan(binsize):
        pass
    else:
        n=np.floor(N/binsize)
    anibins=np.zeros((n+1,))
    for i in range(n+1):
        anibins[i]=np.nanpercentile(ani[mmm],i/n*100)
    return anibins,mmm

#############################################################################
##################### COMPARE CURVE FIT RESULTS #############################
# Compares the results of different curve fits
def compare_fits(reports):

    ''' turns a list of reports into useful output '''
    names=['ID------------------','-SCORE-','--SS---','-medSS-','-sigSS-','-mSSlo-','-mSShi-','Nworse-']
    vari=[]
    for rp in reports:
        vari.append(rp.var)
    vari=np.array(vari)
    for v in np.unique(vari):
        N=np.sum(vari==v)
        full=np.ones((N,len(names)))*float('nan')
        for i in range(N):
            idx=np.where(vari==v)[0][i]
            rp=reports[idx]
            full[i,0]=idx
            a=len(rp.params)
            if rp.stats in [None,{}]:
                continue
            full[i,2]=rp.stats['SS']
            full[i,3]=rp.stats['medSS']
            full[i,4]=rp.stats['sigSS']
            full[i,5]=rp.stats['medSSlo']
            full[i,6]=rp.stats['medSShi']
            full[i,7]=rp.stats['N_worse']

            full[i,1]=(rp.stats['SS']/.2*5+rp.stats['medSS']/.2*15-rp.stats['sigSS']/.58*5-rp.stats['N_worse']/4\
                    -10*(np.abs(rp.stats['medSSlo']-rp.stats['medSS'])+np.abs(rp.stats['medSShi']-rp.stats['medSS'])))-(a)/2

        print('---------------------------------'+str(v)+'-------------------------------------')
        for n in names:
            print(n,end='|')
        print()
        for i in range(N):
            idx=np.where(vari==v)[0][i]
            rp=reports[idx]
            for n in range(len(names)):
                if n==0:
                    prms_str=''
                    for k in rp.params:
                        prms_str=prms_str+k+','
                    print('{0:20}'.format(rp.base+'('+prms_str[0:-1]+')')+str('|'),end='')
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

#########################################################################
######################## FUNCTION GENERATOR #############################
# generate a function for curve fitting
def fxngen(plist,bsfx,ac_=None,bc_=None,cc_=None,dc_=None,ec_=None):
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
        return bsfx(zL,aa,bb,cc,dd,ee)
    return fxn







