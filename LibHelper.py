import numpy as np
import numpy.random as ra


import matplotlib.pyplot as plt
from NiceColours import *

# Nicer names
def NiceSet(Subset):
    Nicer = {'tmd':'TMDi', 'mor':'MOR13', 'tmb':'TMOth'}
    if Subset in Nicer: return Nicer[Subset]
    return Subset

# Pedagogical names
def PedagSet(Subset):
    Nicer = {'tmd':'TM', 'mor':'TM+O', 'tmb':'TM',
             'GMTKN55':'Organic'}
    if Subset in Nicer: return Nicer[Subset]
    return Subset


# Various DFA models

# 1 parameter = PBE0 family (physics constrained)
def DFA1p(c):
    return {'xhf':1-c[0], 'xpbe':c[0], 'cpbe':1.}
# 1 parameter = BLYP family (physics constrained)
def DFA1p_BLYP(c):
    return {'xhf':1-c[0], 'xb':c[0], 'clyp':1.}
# 1 parameter = SCAN0 family (physics constrained)
def DFA1p_SCAN(c):
    return {'xhf':1-c[0], 'xscan':c[0], 'cscan':1.}

# 2 parameter = PBE0 + MP2 (semi-physics constrained)
def DFA2p(c):
    return {'xhf':1-c[0], 'xpbe':c[0], 'cmp2':1-c[1], 'cpbe':c[1]}
# 2 parameter = BLYP + MP2 (semi-physics constrained)
def DFA2p_BLYP(c):
    return {'xhf':1-c[0], 'xb':c[0], 'cmp2':1-c[1], 'clyp':c[1]}
# 2 parameter = SCAN0 + MP2 (semi-physics constrained)
def DFA2p_SCAN(c):
    return {'xhf':1-c[0], 'xscan':c[0], 'cmp2':1-c[1], 'cscan':c[1]}

# XYG family per Stefan's descriptions
def XYG(c):
    if len(c)==1:
        a1 = c
        a6, a7 = a1**2, a1**2
        a2, a3, a4, a5 = 0, 1-a1, 0, 1-a1**2
    elif len(c)==2:
        a1, a6 = tuple(c)
        a2, a3, a4, a5, a7 = 0, 1-a1, 0, 1-a6, a6
    elif len(c)==3:
        a1, a3, a6 = tuple(c)
        a2, a4, a5, a7 = 0., 0., 1-a6, a6
    elif len(c)==4:
        a1, a2, a3, a6 = tuple(c)
        a4, a5, a7 = 0., 1-a6, a6
    elif len(c)==5:
        a1, a2, a3, a5, a6 = tuple(c)
        a4, a7 = 0., a6
    elif len(c)==6:
        a1, a2, a3, a4, a5, a6 = tuple(c)
        a7 = a6
    elif len(c)==7:
        a1, a2, a3, a4, a5, a6, a7 = tuple(c)
    else: return None
    
    return {'xhf':a1, 'xlda':a2, 'xpbe':a3, 
            'clda':a4, 'cpbe':a5, 'cmp2ss':a6, 'cmp2os':a7 }

def XYG_BLYP(c):
    if len(c)==1:
        a1 = c
        a6, a7 = a1**2, a1**2
        a2, a3, a4, a5 = 0, 1-a1, 0, 1-a1**2
    elif len(c)==2:
        a1, a6 = tuple(c)
        a2, a3, a4, a5, a7 = 0, 1-a1, 0, 1-a6, a6
    elif len(c)==3:
        a1, a3, a6 = tuple(c)
        a2, a4, a5, a7 = 0., 0., 1-a6, a6
    elif len(c)==4:
        a1, a2, a3, a6 = tuple(c)
        a4, a5, a7 = 0., 1-a6, a6
    elif len(c)==5:
        a1, a2, a3, a5, a6 = tuple(c)
        a4, a7 = 0., a6
    elif len(c)==6:
        a1, a2, a3, a4, a5, a6 = tuple(c)
        a7 = a6
    elif len(c)==7:
        a1, a2, a3, a4, a5, a6, a7 = tuple(c)
    else: return None
    
    return {'xhf':a1, 'xlda':a2, 'xb':a3, 
            'clda':a4, 'clyp':a5, 'cmp2ss':a6, 'cmp2os':a7 }

def XYG_SCAN(c):
    if len(c)==1:
        a1 = c
        a6, a7 = a1**2, a1**2
        a2, a3, a4, a5 = 0, 1-a1, 0, 1-a1**2
    elif len(c)==2:
        a1, a6 = tuple(c)
        a2, a3, a4, a5, a7 = 0, 1-a1, 0, 1-a6, a6
    elif len(c)==3:
        a1, a3, a6 = tuple(c)
        a2, a4, a5, a7 = 0., 0., 1-a6, a6
    elif len(c)==4:
        a1, a2, a3, a6 = tuple(c)
        a4, a5, a7 = 0., 1-a6, a6
    elif len(c)==5:
        a1, a2, a3, a5, a6 = tuple(c)
        a4, a7 = 0., a6
    elif len(c)==6:
        a1, a2, a3, a4, a5, a6 = tuple(c)
        a7 = a6
    elif len(c)==7:
        a1, a2, a3, a4, a5, a6, a7 = tuple(c)
    else: return None
    
    return {'xhf':a1, 'xlda':a2, 'xscan':a3, 
            'clda':a4, 'cscan':a5, 'cmp2ss':a6, 'cmp2os':a7 }




# Optimize alpha
def Optalpha(ERef, EPBE, EHFc, return_all=False):
    a = np.linspace(-2., 3., 301)
    Erra = np.mean(np.abs(EPBE[:,None]*(1.-a[None,:])
                          + EHFc[:,None]*a[None,:]
                          - ERef[:,None]), axis=0)
    q = np.argmin( Erra )

    if return_all:
        return a[q], Erra[q], a, Erra
    else:
        return a[q], Erra[q]

# Optimize over random subsets
def GetTraining(ERef, EPBE, EHFc,
                NRun=50, Percent=80):
    N = len(ERef)
    NT = int(np.round(N*Percent/100.))
    if NT>=N:
        return Optalpha(ERef, EPBE, EHFc)

    alpha0List = [None]*NRun
    Erra0List = [None]*NRun
    for Run in range(NRun):
        kk = ra.choice(range(N), NT, replace=False)
        alpha0, Erra0 = Optalpha(ERef[kk], EPBE[kk], EHFc[kk])
        alpha0List[Run] = alpha0
        Erra0List[Run] = Erra0
    return np.array(alpha0List), np.array(Erra0List)
    

###############################################################
# Plotting helper
###############################################################

# Add a comparison chart like Stefan does
def Comparison(ax, DB, SetList,
               # Defaults to XYG with 4 parameters
               DFA=XYG, c0=[0.5,0.5,0.5,0.5],
               Values=True,
               ErrMin = 0.01, # Minimal error
               ):
    NSet = len(SetList)
    Err = np.zeros((NSet,NSet))
    for K1, Set1 in enumerate(SetList):
        c, _ = DB.OptGeneral(Set1, DFA, c0)
        for K2, Set2 in enumerate(SetList):
            Err[K1,K2] = DB.GetMAD(Set2, DFA(c))
            
    DErr = (Err+ErrMin) / (np.diag(Err)+ErrMin)[None,:]

    if Values:
        for K1 in range(NSet):
            for K2 in range(NSet):
                ErrVal = Err[K1,K2]
                ErrCol = DErr[K1,K2]
                if ErrVal<10: ErrTxt = "%.2f"%(ErrVal)
                elif ErrVal<100: ErrTxt = "%.1f"%(ErrVal)
                else: ErrTxt = "%.0f"%(ErrVal)

                Coltxt, Colbdr = 'w', 'k'
                if ErrCol>2.0: Coltxt, Colbdr = 'k', 'w'

                AddBorder(
                    ax.text(K1, K2, ErrTxt,
                            color=Coltxt,
                            fontsize=8,
                            rotation=45,
                            horizontalalignment="center",
                            verticalalignment="center",
                            ),
                    w=1, fg=Colbdr,
                )
    else:
        for V in (1., 1.2, 1.4, 1.6, 1.8, 2.0):
            J0 = np.unravel_index(np.argmin(np.abs(DErr-V), axis=None),
                                  DErr.shape)
            K1, K2 = tuple(J0)
            Rem = np.abs(DErr[K1,K2]-V)
            if V<1.5: Coltxt, Colbdr = 'w', 'k'
            else:  Coltxt, Colbdr = 'k', 'w'

            if V<2.: Txt="%.1f"%(V)
            else: Txt="2+"
            if Rem<0.05 or V==2.:
                AddBorder(
                    ax.text(K1, K2, Txt,
                            color=Coltxt,
                            fontsize=8,
                            rotation=45,
                            horizontalalignment="center",
                            verticalalignment="center",
                            ),
                    w=1, fg=Colbdr,
                )
    
    
    ax.imshow(DErr.T, vmin=1., vmax=2, )
    ax.set_xticks(range(NSet), [NiceSet(X) for X in SetList],
                  fontsize=8,
                  rotation=45)
    ax.set_yticks(range(NSet), [NiceSet(X) for X in SetList],
                  fontsize=8,
                  rotation=45)
   
def simple_beeswarm(y, nbins=None, ylo=None, yhi=None):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    y = np.asarray(y)
    if nbins is None:
        nbins = len(y) // 6

    # Get upper bounds of bins
    x = np.zeros(len(y))
    if ylo is None: ylo = np.min(y)
    if yhi is None: yhi = np.max(y)
    dy = (yhi - ylo) / nbins
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = 1 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x
