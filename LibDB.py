import numpy as np
import numpy.random as ra
import scipy.optimize as opt
import scipy.linalg as la

import os

#########################################################################
# Make the code deterministic by pre-generating
# random c0 values by pseudo-random numbers with a seed

ra.seed(123456)
c0_Pseudo = {}
c0_MaxPRandom = 50
for NP in range(1,12):
    c0_Pseudo[NP] = ra.random(size=(c0_MaxPRandom,NP))

#########################################################################
# Key DFAs
HF   = {'xhf': 1.,}
MP2  = {'xhf': 1., 'cmp2': 1.}
PBE  = {'xpbe': 1., 'cpbe': 1.,}
PBE0 = {'xhf': 0.25, 'xpbe': 0.75, 'cpbe': 1.,}

DB_TM151  = ["tmd", "tmb", "mor"]
DB_TMC151 = ["tmd", "tmb", "mor"]
DB_GMTKN55 = ['ACONF', 'ADIM6', 'AHB21', 'AL2X6', 'ALK8', 'ALKBDE10',
              'Amino20x4', 'BH76', 'BH76RC', 'BHDIV10', 'BHPERI',
              'BHROT27', 'BSR36', 'BUT14DIOL', 'C60ISO', 'CARBHB12',
              'CDIE20', 'CHB6', 'DARC', 'DC13', 'DIPCS10', 'FH51',
              'G21EA', 'G21IP', 'G2RC', 'HAL59', 'HEAVY28',
              'HEAVYSB11', 'ICONF', 'IDISP', 'IL16', 'INV24',
              'ISO34', 'ISOL24', 'MB16-43', 'MCONF', 'NBPRC',
              'PA26', 'PArel', 'PCONF21', 'PNICO23', 'PX13',
              'RC21', 'RG18', 'RSE43', 'S22', 'S66', 'SCONF',
              'SIE4x4', 'TAUT15', 'UPU23', 'W4-11',
              'WATER27', 'WCPT18', 'YBDE18']

# Only the covalent interactions in GMTKN55
DB_GMTKN55_CI = [ 'AL2X6', 'ALK8', 'ALKBDE10', 'BH76', 'BH76RC',
                  'BHDIV10', 'BHPERI', 'BHROT27', 'BSR36',
                  'C60ISO', 'CDIE20', 'DARC', 'DC13', 'DIPCS10',
                  'FH51', 'G21EA', 'G21IP', 'G2RC', 'HEAVYSB11',
                  'INV24', 'ISO34', 'ISOL24', 'MB16-43', 'NBPRC',
                  'PA26', 'PArel', 'PX13', 'RC21', 'RSE43',
                  'SIE4x4', 'TAUT15', 'W4-11', 'WCPT18', 'YBDE18' ]

#########################################################################
# Helper functions
#########################################################################

def NiceArr(X):
    return ", ".join(["%7.3f"%(x) for x in X])

def ReadEnergies(Dir, ID):
    X = np.loadtxt(Dir + ID)
    return X
    

def ReadReactions(Dir):
    F1 = open(Dir + "/reacs.dat")
    Lines1 = list(F1)
    F1.close()

    F2 = open(Dir + "/coeffs.dat")
    Lines2 = list(F2)
    F2.close()

    N1, N2 = len(Lines1), len(Lines2)
    
    if not(N1==N2):
        print(Lines1)
        print(Lines2)
        print("Lengths are different in "+Dir)
        quit()
    #else:
    #    print("Database in %s has %d reactions"%(TopDir, N1))

    Reactions = [None]*N1
    for K, (L1, L2) in enumerate(zip(Lines1, Lines2)):
        X1 = L1.split()
        X2 = L2.split()

        Reactions[K] = {}
        for x1, x2 in zip(X1, X2):
            Reactions[K][x1] = float(x2)

    return Reactions

# texify a DFA
def texify(DFA):
    texMap = {
        'xhf': 'E_{\\xrm}^{\\HF}',
        'xlda': 'E_{\\xrm}^{\\LDA}',
        'xb': 'E_{\\xrm}^{\\text{B88}}',
        'xpbe': 'E_{\\xrm}^{\\text{PBE}}',
        'clda': 'E_{\\crm}^{\\LDA}',
        'clyp': 'E_{\\crm}^{\\text{LYP}}',
        'cpbe': 'E_{\\crm}^{\\text{PBE}}',
        'cmp2ss': 'E_{\\crm}^{\\MP2_{\\text{ss}}}',
        'cmp2os': 'E_{\\crm}^{\\MP2_{\\text{os}}}',
    }

    Str = ""
    for K, ID in enumerate(
            ('xhf', 'xlda', 'xb', 'xpbe', 
             'clda', 'clyp', 'cpbe', 'cmp2ss', 'cmp2os') ):
        if not(ID in DFA): continue

        a = DFA[ID]
        tex = texMap[ID]

        if K==0 and a>0.: Pre = "%.3f"%(a)
        elif a>0.: Pre = " + %.3f"%(a)
        elif a==0.: continue
        else: Pre = " - %.3f"%(-a)

        Str += Pre+tex
    return Str
        
#########################################################################
# DFA functions
#########################################################################

# PBE0 family
def DFA_PBE0(alpha):
    return {'xhf':alpha, 'xpbe':1.-alpha, 'cpbe':1.}

# XYG family per Stefan's descriptions
def DFA_XYG(c, xGGA='xpbe', cGGA='cpbe'):
    Nc = len(np.atleast_1d(c))
    if Nc==1:
        a1 = c
        a6, a7 = a1**2, a1**2
        a2, a3, a4, a5 = 0, 1-a1, 0, 1-a1**2
    elif Nc==2:
        a1, a6 = tuple(c)
        a2, a3, a4, a5, a7 = 0, 1-a1, 0, 1-a6, a6
    elif Nc==3:
        a1, a3, a6 = tuple(c)
        a2, a4, a5, a7 = 0., 0., 1-a6, a6
    elif Nc==4:
        a1, a2, a3, a6 = tuple(c)
        a4, a5, a7 = 0., 1-a6, a6
    elif Nc==5:
        a1, a2, a3, a5, a6 = tuple(c)
        a4, a7 = 0., a6
    elif Nc==6:
        a1, a2, a3, a4, a5, a6 = tuple(c)
        a7 = a6
    elif Nc==7:
        a1, a2, a3, a4, a5, a6, a7 = tuple(c)
    else: return None
    
    return {'xhf':a1, 'xlda':a2, xGGA:a3, 
            'clda':a4, cGGA:a5, 'cmp2ss':a6, 'cmp2os':a7 }

# XYG other variants
def DFA_XYG_BLYP(c):
    return DFA_XYG(c, xGGA='xb', cGGA='clyp')

def DFA_XYG_PBE(c):
    return DFA_XYG(c, xGGA='xpbe', cGGA='cpbe')

# XYG other variants
def DFA_XYG_BPBE(c):
    return DFA_XYG(c, xGGA='xb', cGGA='cpbe')
    
def DFA_XYG_SCAN(c):
    return DFA_XYG(c, xGGA='xscan', cGGA='cscan')
 
def DFA_XYG_BSCAN(c):
    return DFA_XYG(c, xGGA='xb', cGGA='cscan')
    
def DFA_XYG_PBESCAN(c):
    return DFA_XYG(c, xGGA='xpbe', cGGA='cscan')


# XYG9 = XYG7 [BLYP] + xpbe + cpbe
def DFA_XYG9(c):
    Nc = len(np.atleast_1d(c))
    if Nc==9:
        a1, a2, a3, a4, a5, a6, a7, a8, a9 = tuple(c)
    else: return None
    
    return {'xhf':a1, 'xlda':a2, 'xb':a3, 
            'clda':a4, 'clyp':a5, 'cmp2ss':a6, 'cmp2os':a7, 'xpbe':a8, 'cpbe':a9 }

# XYG11 = XYG7 [BLYP] + xpbe + cpbe + xscan + scan
def DFA_XYG11(c):
    Nc = len(np.atleast_1d(c))
    if Nc==11:
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = tuple(c)
    else: return None
    
    return {'xhf':a1, 'xlda':a2, 'xb':a3, 
            'clda':a4, 'clyp':a5, 'cmp2ss':a6, 'cmp2os':a7, 'xpbe':a8, 'cpbe':a9, 'xscan':a10, 'cscan':a11}


#########################################################################
#helper function for pseudo_random_generation:

def my_pseudo_random(len_c0, some_parameter):
    a = 1664525
    c = 1013904223
    m = 2**32  # modulus value
    x = some_parameter  # seed
    pseudo_random_values = []

    for _ in range(len_c0):
        x = (a * x + c) % m
        normalized_value = x / m  # normalize to [0, 1)
        
        # Make sure values are between 0 and 1
        if normalized_value == 0:
            normalized_value += 1e-10  # some small value
        elif normalized_value == 1:
            normalized_value -= 1e-10  # some small value
        
        pseudo_random_values.append(normalized_value)
        
    return pseudo_random_values
 
#########################################################################
# Reaction database
#########################################################################

class DFADB:
    # Initialise the class
    def __init__(self, Rebuild=False,
                 DB=None, ID=None):
        self.CacheName = "_LibDB-Cache_v2.npz"

        self.xType = ['xlda', 'xpbe', 'xb', 'xscan', 'xhf',]
        self.cType = ['clda', 'clyp', 'cpbe', 'clyp', 'scan',
                      'cmp2', 'cmp2os', 'cmp2ss',]

        self.FullCache = {'Reactions':{},
                          'NReactions':{},
                          'RefEnergies':{},
                          'Energies':{},}
        self.FullReactions = {}
        self.AllEnNames = {}
        
        if not(Rebuild):
            try:
                X = np.load(self.CacheName, allow_pickle=True)
                self.FullCache = X['FullCache'][()]
                self.FullReactions = X['FullReactions'][()]
                self.AllEnNames = X['AllEnNames'][()]
            except:
                print("Error in cache - rebuilding")
        else: print("Rebuilding cache")

        # If asked, update from sub-directory DB with ID = DB by default
        if not(DB is None): self.Update(DB, ID)

    # Save the cache
    def UpdateCache(self):
        np.savez(self.CacheName,
                 FullCache=self.FullCache,
                 FullReactions=self.FullReactions,
                 AllEnNames=self.AllEnNames,
                 )

    # Read the information from a sub-directory and update the database
    def Update(self, DB, ID=None):
        if ID is None: ID = DB

        if ID in self.FullCache['RefEnergies']:
            self.AllEnNames = list(self.FullCache['Energies'][ID])
            return ID

        TD = "./reaction_numbers/%s/"%(DB)

        if not(os.path.isdir(TD)):
            print("Warning! \"%s\" does not exist - check case")
            return None
        print("Reading %s"%(TD))

        Reactions = ReadReactions(TD)
        ReactionID = [ ID+":%d"%(k+1) for k in range(len(Reactions)) ]
        for k, RID in enumerate(ReactionID):
            self.FullReactions[RID] = Reactions[k]

        self.FullCache['Reactions'][ID] = ReactionID
        self.FullCache['NReactions'][ID] = len(self.FullCache['Reactions'][ID])
        self.FullCache['RefEnergies'][ID] = ReadEnergies(TD, "refs.dat")
        self.FullCache['Energies'][ID] = {}
        for Q in ['hf',] + self.xType + self.cType:
            self.FullCache['Energies'][ID][Q.lower()] \
                = ReadEnergies(TD, Q+".dat")

        # The mean-field energy
        self.FullCache['Energies'][ID]['mf'] \
            = self.FullCache['Energies'][ID]['hf'] \
            - self.FullCache['Energies'][ID]['xhf']

        # Alias for cscan
        self.FullCache['Energies'][ID]['cscan'] \
            = self.FullCache['Energies'][ID]['scan']

        self.AllEnNames = list(self.FullCache['Energies'][ID])

        self.UpdateCache()

        return ID

    # Create a new set as an alias to an existing set
    def Alias(self, NewID, OldID):
        return self.Combine(NewID, [OldID,])

    # Combine subsets to make a new database
    # NewID = Name of new database
    # IDList = (ID1, ID2, ...)
    def Combine(self, NewID, IDList, Prune=False):
        # Do not recompute if already exists
        #if NewID in self.FullCache['RefEnergies']: return NewID

        
        self.FullCache['Reactions'][NewID] = []
        self.FullCache['NReactions'][NewID] = 0
        self.FullCache['RefEnergies'][NewID] = []
        self.FullCache['Energies'][NewID] = {}
        for ID in IDList:
            if not(ID) in self.FullCache['Reactions']:
                self.Update(ID)
                
            self.FullCache['Reactions'][NewID] += \
                self.FullCache['Reactions'][ID]
            self.FullCache['NReactions'][NewID] += \
                self.FullCache['NReactions'][ID]
            if len(self.FullCache['RefEnergies'][NewID])==0:
                self.FullCache['RefEnergies'][NewID] \
                    = self.FullCache['RefEnergies'][ID]*1.
                for Q in self.AllEnNames:
                    self.FullCache['Energies'][NewID][Q] \
                        = self.FullCache['Energies'][ID][Q]*1.
            else:
                self.FullCache['RefEnergies'][NewID] = \
                    np.hstack((self.FullCache['RefEnergies'][NewID],
                                 self.FullCache['RefEnergies'][ID]*1.))
                for Q in self.AllEnNames:
                    self.FullCache['Energies'][NewID][Q] =  \
                        np.hstack((self.FullCache['Energies'][NewID][Q],
                                   self.FullCache['Energies'][ID][Q]*1.))

        if Prune: self.Prune(NewID)
        
        self.UpdateCache()
        return NewID

    # Remove duplicate entries
    def Prune(self, ID):
        Reactions = self.FullCache['Reactions'][ID]
        NReactions = self.FullCache['NReactions'][ID]
        RefEns = self.FullCache['RefEnergies'][ID]*1.

        PruneList = []
        for K, R in enumerate(Reactions):
            for KP in range(K+1, NReactions):
                if Reactions[K]==Reactions[KP]:
                    PruneList += [KP,]

        PruneList = set(PruneList)

        KFinal = []
        for K in range(NReactions):
            if not(K in PruneList): KFinal += [K,]

        self.FullCache['Reactions'][ID] \
            = [Reactions[KP] for KP in KFinal]
        self.FullCache['Reactions'][ID] = len(KFinal)
        self.FullCache['RefEnergies'][ID] = RefEns[KFinal]
    
        print("Pruning %4d from %4d to leave %4d"\
              %(len(PruneList), NReactions,
                self.FullCache['Reactions'][ID]))
        

    
    # Extract a special set
    # If ID is DietNNN or P30-NN or PoisonNN it will automatically
    #     extract details
    # Otherwise, can specify a list like
    #     {'Subset1':[1,5,6,...], 'Subset2':[]}
    #     where the numbering starts at 1 (Zero=False, default)
    #     or 0 (Zero=True) and [] extracts the whole subset
    def Extract(self, ID, List=None, Zero=False):
        #if ID in self.FullCache['RefEnergies']: return ID

        self.Combine("GMTKN55", DB_GMTKN55)
        self.Combine("TMC151" , DB_TMC151 )

        if ID[:3].lower() in ('die', 'p30', 'poi', 'tra'):
            List = {}
            Zero=False

            # Handle diet set
            if ID[:3].lower()=="die":
                N = int(ID[4:])

                import yaml
                F = open("reaction_numbers/diet/%d.txt"%(N))
                X = yaml.load(F, Loader = yaml.UnsafeLoader)
                
                for IDS in X['Systems']:
                    List[IDS] = X['Systems'][IDS][1]
            elif ID[:4].lower() in ("tran", "t100"):
                F = open("BenchmarkSets/%s.txt"%(ID))

                List = {}
                N = 0
                for L in F:
                    X = L.split()
                    if len(X)==3:
                        N += 1
                        Y = X[0].split(':')
                        BS, K = tuple(Y)
                        if not(BS in List): List[BS] = []
                        List[BS] += [int(K),]
            else:
                N = 30
                
                if ID[:6].lower()=="poison": Max = int(ID[6:])
                else: Max = int(ID[4:])

                F = open("poison/Systems_P30-%d.txt"%(Max))
                for L in F:
                    IDS, H = L.split(':')
                    if IDS in List: List[IDS] += [int(H)]
                    else: List[IDS] = [int(H)]
        elif List is None:
            return None
        else:
            # Get the length
            N = 0
            for S in List:
                # An empty list is converted to a full list
                if len(List[S])==0:
                    NEntry=len(self.FullCache['RefEnergies'][S])
                    if Zero: List[S] = list(range(NEntry))
                    else: List[S] = list(range(1,NEntry+1))
                # A negative first entry gets everything out
                if List[S][0]<0:
                    NEntry=len(self.FullCache['RefEnergies'][S])
                    if Zero: TList = list(range(NEntry))
                    else: TList = list(range(1,NEntry+1))
                    
                    for E in List[S]: TList.remove(np.abs(E))

                    List[S] = TList
                    

                N += len(List[S])

        self.FullCache['Reactions'][ID] = [None]*N
        self.FullCache['NReactions'][ID] = N
        self.FullCache['RefEnergies'][ID] = np.zeros((N,))
        self.FullCache['Energies'][ID] = {}
        for Q in self.FullCache['Energies']['GMTKN55']:
            self.FullCache['Energies'][ID][Q] = np.zeros((N,))

        K = 0
        for Set in List:
            for H in List[Set]:
                if Zero: J = H
                else: J = H-1
                self.FullCache['Reactions'][ID][K] = \
                    self.FullCache['Reactions'][Set][J]
                self.FullCache['RefEnergies'][ID][K] = \
                    self.FullCache['RefEnergies'][Set][J]
                for Q in self.FullCache['Energies']['GMTKN55']:
                    self.FullCache['Energies'][ID][Q][K] = \
                        self.FullCache['Energies'][Set][Q][J]

                K += 1

        return ID

    # Find the size of a set
    def Size(self, ID):
        return self.FullCache['NReactions'][ID]
        
    # Report on the subset
    # Long=True gives detailed reaction information
    def GetInfo(self, ID, Long = False):
        if ID in self.FullCache['Reactions']:
            print("Database %s has %d reactions"\
                  %(ID, self.FullCache['NReactions'][ID]))
            if Long:
                for K, R in enumerate(self.FullCache['Reactions'][ID]):
                    ER = self.FullCache['RefEnergies'][ID][K]
                    V = []
                    for X in R:
                        N = R[X]
                        V += ["%.1f %s"%(N, X)]
                    print("%7.2f from "%(ER) + " + ".join(V))

    # Get the reactions
    def GetMolecules(self, ID, kk=None):
        AllMols = []
        if kk is None: kk = range(len(self.FullCache['Reactions'][ID]))
        for k in kk:
            RID=self.FullCache['Reactions'][ID][k]
            SetID = RID.split(':')[0]
            Reaction = self.FullReactions[RID]
            for M in Reaction:
                W = Reaction[M]
                FullID = SetID + ":" + M
                AllMols += [(SetID, M, FullID, W)]

        return AllMols
    
    # Get the reference energies for set ID
    def RefEnergies(self, ID):
        return self.FullCache['RefEnergies'][ID]*1.

    # Get component Q (e.g. xpbe, cmp2os)
    def Energies(self, ID, Q):
        return self.FullCache['Energies'][ID][Q]*1.
    
    # Get energies from a DFA
    # Use database ID
    # DFA specifies the combination of energies, e.g.
    #     HF   = {'xhf': 1.,}
    #     PBE  = {'xpbe': 1., 'cpbe': 1.,}
    #     PBE0 = {'xhf': 0.25, 'xpbe': 0.75, 'cpbe': 1.,}    
    def GetEnergies(self, ID, DFA, WithMF=True, kk=None):
        # By default it will include the MF energy in full
        if WithMF:
            if kk is None: E0 = self.FullCache['Energies'][ID]['mf']*1.
            else: E0 = self.FullCache['Energies'][ID]['mf'][kk]
        else: E0 = 0.

        # Add the combination
        for Q in DFA:
            W = DFA[Q]
            if kk is None: E0 += W * self.FullCache['Energies'][ID][Q.lower()]
            else: E0 += W * self.FullCache['Energies'][ID][Q.lower()][kk]

        return E0

    # Get the errors
    def GetErrors(self, ID, DFA, WithMF=True, kk=None):
        if kk is None: ERef = self.FullCache['RefEnergies'][ID]
        else: ERef = self.FullCache['RefEnergies'][ID][kk]
        return self.GetEnergies(ID, DFA,
                                WithMF=WithMF, kk=kk)\
            - ERef

    # Get the MAD on the set
    def GetMAD(self, ID, DFA, WithMF=True, kk=None):
        return np.mean(np.abs(self.GetErrors(ID, DFA,
                                             WithMF=WithMF, kk=kk)))

    # Get a weighted MAD on the set
    #    WSet = {'ID':val, ...} weights by subset (defaults to one otherwise)
    #    W = [0.1, 0.2, 0.3, ...] weight by entry (must be right length)
    #    return_W returns a weight vector of the same length as the full set
    def GetWMAD(self, ID, DFA, WithMF=True,
                WSet = None, W = None,
                return_W = False,
                ):
        Err = self.GetErrors(ID, DFA,
                             WithMF=WithMF, kk=kk)
        
        if not(WSet is None):
            W = 0.*Err
            for K, R in enumerate(self.FullCache['Reactions'][ID]):
                RID = R.split(':')[0]

                if RID in WSet: W[K] = WSet[RID]
                else: W[K] = 1.
                
        elif not(W is None):
            if not(len(W) == len(Err)):
                print("Weights are wrong length")
                return None
        else:
            W = 0.*Err + 1/len(Err)

        if return_W: return W
        
        return np.sum(W*np.abs(Err))

    # General optimization using RMSD as a guess
    # General optimization with initial seeds 
    def OptGeneral_Quick(self, ID, DFAModel, c0, kk=None,
                         eta = 1e-5):
        # Compute the RMSD solution
        Y = self.FullCache['RefEnergies'][ID][kk].reshape((-1,))
        NOpt = len(c0)
        NSet = len(Y)

        E = np.eye(NOpt)

        X = np.zeros((NSet, NOpt))
        for k in range(NOpt):
            X[:,k] = self.GetEnergies(ID, DFAModel(E[k,:]),
                                      WithMF=True, kk=kk)

        A = (X.T).dot(X)
        A += eta*np.trace(A)/NOpt * np.eye(NOpt)
        B = (X.T).dot(Y)
        c0 = la.solve(A, B) # We will start from the RMSD solution

        
        def Err(c):
            return self.GetMAD(ID, DFAModel(c), kk=kk)

        res = opt.minimize(Err, x0=c0)
        c = res.x

        return c, Err(c)

    # Alias for Quick used in older routines
    def OptGeneral_RMSD(self, ID, DFAModel, c0, **kwargs):
        return self.OptGeneral_Quick(ID, DFAModel, c0,
                                     **kwargs)
    

    # Iteratively re-weighted least squares
    #  see https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares
    def OptGeneral_IWLS(self, ID, DFAModel, c0, kk=None,
                        eta=1e-5, delta=1e-4, MaxStep=100):

        def Err(c):
            return self.GetMAD(ID, DFAModel(c), kk=kk)

        # Do a simple optimize for one parameter
        if len(c0)==1:
            res = opt.minimize(Err, c0)
            return res.x, res.fun
        
        # Compute the RMSD solution
        YRaw = self.FullCache['RefEnergies'][ID][kk].reshape((-1,))
        NOpt = len(c0)
        NSet = len(YRaw)

        YZero = self.GetEnergies(ID, DFAModel([0.,]*NOpt), kk=kk)
        Y = YRaw - YZero
        
                
        E = np.eye(NOpt)

        X = np.zeros((NSet, NOpt))
        for k in range(NOpt):
            X[:,k] = self.GetEnergies(ID, DFAModel(E[k,:]), kk=kk) - YZero

        if MaxStep == 0 \
           or (np.sum(np.abs(c0))==0.): # If RMSD or all zeros
            A = np.einsum('ij,ik->jk', X, X)
            A += eta * np.eye(NOpt)
            B = np.einsum('ij,i', X, Y)
            c = la.solve(A, B) # We will start from the RMSD solution
        else:
            c = np.array(c0)
            
        ErrBest, cBest = Err(c), c
        for step in range(MaxStep):
            W = 1./np.maximum(delta, np.abs(Y - X.dot(c)))
            # Ensure the minimum is one
            W /= W.min()
            
            A = np.einsum('ij,i,ik->jk', X, W, X)
            A += eta * np.eye(NOpt)
            B = np.einsum('ij,i,i', X, W, Y)
            c = la.solve(A, B) # We will start from the RMSD solution


            Err_c = Err(c)
            if (np.abs(Err_c-ErrBest)<1e-4) \
               or (Err_c-ErrBest)>1.:
                break
            
            if Err_c<ErrBest:
                ErrBest, cBest = Err_c, c

        return cBest, Err(cBest)

    # General optimization with initial seeds 
    def OptGeneral_seed(self, ID, DFAModel, c0, kk=None):
        def Err(c):
            return self.GetMAD(ID, DFAModel(c), kk=kk)

        res = opt.minimize(Err, x0=c0)
        c = res.x

        return c, Err(c)

    # General optimization with multiple seeds
    def OptGeneral_Multi(self, ID, DFAModel, c0, kk=None,
                         N_extra_seeds=10):
        #
        static_seeds = [0.125, 0.25, 0.75, 1.0]
        best_result = None
        best_min_value = float('inf')

        # Static seeds opt
        for seed in static_seeds:
            c_seed = np.ones((len(c0),)) * seed
            result, min_value = self.OptGeneral_seed(ID, DFAModel,
                                                     c_seed, kk=kk)

            if min_value < best_min_value:
                best_min_value = min_value
                best_result = result

        # Randomly generated seeds opt
        if N_extra_seeds > 0:
            for i0 in range(N_extra_seeds):
                # Old deterministic code using pseudo-randoms
                c_seed = c0_Pseudo[len(c0)][(i0%c0_MaxPRandom),:]
                
                
                result, min_value = self.OptGeneral_seed(ID, DFAModel,
                                                         c_seed, kk=kk)

                if min_value < best_min_value:
                    best_min_value = min_value
                    best_result = result

        return best_result, best_min_value

    # Besf ot optimization with multiple seeds and IWLS
    def OptGeneral_Best(self, ID, DFAModel, c0, kk=None,
                        **kwargs):
        cI, ErrI = self.OptGeneral_IWLS(ID, DFAModel, c0, kk,
                                        **kwargs)
        cM, ErrM = self.OptGeneral_IWLS(ID, DFAModel, c0, kk,
                                        **kwargs)

        if ErrI<ErrM: return cI, ErrI
        else: return cM, ErrM

    # Default optimization
    def OptGeneral(self, ID, DFAModel, c0, kk=None,
                   **kwargs):
        return self.OptGeneral_Best(ID, DFAModel, c0, kk,
                                    **kwargs)


    # PBE0 optimization
    # Note, NParam is never used
    def OptPBE0(self, ID, kk=None, **kwargs):
        ErrBest = 1e10
        def Err(alpha):
            return self.GetMAD(ID, DFA_PBE0(alpha), kk=kk)

        res = opt.minimize_scalar(Err, bounds=[-0.25,1.25])
        c = res.x

        return c, Err(c)

    # XYG optimization
    # Quick
    def OptXYGQuick(self, ID, NParam=7, DFAProps=None, kk=None):
        return self.OptXYG(ID, NParam, DFAProps, kk=kk, RMSD = True)

    # Safe
    def OptXYG(self, ID, NParam=7, DFAProps=None, kk=None,
               RMSD = False, N_extra_seeds=5, DiffEvol=True):
        if NParam==1: c0 = [ 0.93 ]
        elif NParam==2: c0 = [ 0.68, 0.55 ]
        elif NParam==3: c0 = [ 0.68, 0.31, 0.55 ]
        elif NParam==4: c0 = [ 0.70, -0.02, 0.33, 0.58 ]
        elif NParam==5: c0 = [ 0.78, -0.11, 0.32, 0.50, 0.63 ]
        elif NParam==6: c0 = [ 0.79, 0.09, 0.10, 0.31, 0.12, 0.62 ]
        elif NParam==7: c0 = [ 0.79, 0.09, 0.10, 0.31, 0.12, 0.59, 0.64 ]
        #c0 = 1/NParam * np.ones((NParam,))

        if DFAProps is None: DFA = DFA_XYG
        elif str(DFAProps).upper()=='BLYP': DFA = DFA_XYG_BLYP
        elif str(DFAProps).upper()=='BPBE': DFA = DFA_XYG_BPBE
        else: DFA = DFA_XYG
        
        def Err(c):
            return self.GetMAD(ID, DFA(c), kk=kk)

        return self.OptGeneral_IWLS(ID, DFA, c0, kk=kk)            

    # Elemental analysis (slow the first time)
    def ElementalAnalysis(self, ID=None, kk=None, Count=True):
        if ID is None:
            for ID in self.FullCache['Reactions']:
                self.ElementalAnalysis(ID)

        if not(ID in self.FullCache['Reactions']):
            print("%s is not in the reactions:"%(ID))
            print(self.FullCache['Reactions'])
            return None

        try:
            X = np.load("_LibDB-ElCache.npz", allow_pickle=True)
            self.Elements = X['Elements'][()]
        except:
            self.Elements = {}
            
        TD = None
        for B in ("GMTKN55", "TMC151"):
            TD_ = "./structures/%s/%s/"%(B, ID)
            if os.path.isdir(TD_): TD = TD_

        AllMols = self.GetMolecules(ID, kk=kk)
        FullIDList = {}
        for Set, Mol, FullID, W in AllMols:
            if W<=0: continue

            if (FullID in self.Elements): continue
            print(FullID, FullID in self.Elements)

            try:
                F = open(TD+"/%s/struc.xyz"%(Mol))
                N = int(F.readline())
                F.readline()
                Els = {}
                for L in F:
                    El = L.split()[0].title()
                    if not(El in Els): Els[El]=1
                    else: Els[El]+=1
                F.close()
                print("Read %s"%(FullID))
            except:
                print(FullID in self.Elements)
                print("Skipping %s in %s"%(FullID, ID))
                Els = {}

            self.Elements[FullID] = Els

            np.savez("_LibDB-ElCache.npz", Elements=self.Elements)

        np.savez("_LibDB-ElCache.npz", Elements=self.Elements)
        
        if Count:
            if not(kk is None):
                print("Note, kk is ignored with Count=True")
            ElID = {}
            for FullID in FullIDList:
                W = FullIDList[FullID]
                for El in self.Elements[FullID]:
                    if not(El in ElID): ElID[El] = W
                    else: ElID[El] += W
            return ElID
        else:
            NReact = 0
            NIn = {}

            if kk is None:
                kk = range(len(self.FullCache['Reactions'][ID]))
            for k in kk:
                RID=self.FullCache['Reactions'][ID][k]
                NReact += 1
                SID = RID.split(':')[0]
                React = self.FullReactions[RID]
                for Mol in React:
                    W = React[Mol]
                    if W<=0: continue
                
                    FullID = SID+":"+Mol

                    for El in self.Elements[FullID]:
                        if not(El in NIn): NIn[El]=1
                        else: NIn[El] += 1
            fIn = { El:NIn[El]/NReact for El in NIn }
            return NIn, fIn
                    
        return ElID
                
if __name__ == "__main__":
    DB = DFADB(Rebuild = True) # Make sure to set to false normally
    DB.Combine("GMTKN55", DB_GMTKN55)
    DB.Combine("TMC151" , DB_TMC151)

    def DoReport(DB, Subset):
        print("%-6s %6.2f"%("HF"  , DB.GetMAD(Subset, HF  )))
        print("%-6s %6.2f"%("MP2" , DB.GetMAD(Subset, MP2 )))
        print("%-6s %6.2f"%("PBE" , DB.GetMAD(Subset, PBE )))
        print("%-6s %6.2f"%("PBE0", DB.GetMAD(Subset, PBE0)))

        MinErr = DB.GetMAD(Subset, PBE)
        for alpha in np.linspace(0, 1, 101):
            PBE_a = {'xhf':alpha, 'xpbe':1-alpha, 'cpbe':1.}
            Err_a = DB.GetMAD(Subset, PBE_a)
            if (Err_a < MinErr):
                alpha0 = alpha
                MinErr = Err_a
        
        print("%6.2f %6.2f"%(alpha0, MinErr))

    
    
    for Subset in ("tmd", "tmb", "mor"):
        print("="*72)
        DB.Update(Subset)
        DB.GetInfo(Subset)
        print("="*72)
    
        DoReport(DB, Subset)
    
    print("="*72)
    DB.Combine("TMC151", DB_TMC151)
    DB.GetInfo("TMC151")
    print("="*72)
    DoReport(DB, "TMC151")

    DB.Extract("P30-5")
    DB.Extract("P30-10")
    DB.Combine("P60", ["P30-5", "P30-10"])
    DB.Prune("P60")

    if True:
        print("="*72)
        print("Preparing elemental cache")
        print("="*72)

        for ID in list(DB_GMTKN55) + list(DB_TMC151):
            DB.ElementalAnalysis(ID)
