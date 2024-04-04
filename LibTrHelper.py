from LibDB import *

import numpy.random as ra

# Get a suffix
def Suffix(NParam, NSample, ErrMode, FullSets, DFAProps,
           pError=1, Special=""):
    Suff = "%d_%03d_%s"%(NParam, NSample, ErrMode[0].upper())
    if not(FullSets): Suff+="_RedSet"
    if not(DFAProps is None):
        if str(DFAProps).upper()=='BLYP': Suff+="_blyp"
        elif str(DFAProps).upper()=='BPBE': Suff+="_bpbe"

    if not(pError==1): Suff += "_pError%d"%(pError)
        
    return Suff + Special

# Use to define the error function
def DefaultErrFn(E, EB, ID):
    return (E+0.1)/(EB+0.1)

# Use to define the error function
def EnergyErrFn(E, EB, ID):
    return E - EB

# WTMAD-2-like error function
class WTMAD2:
    def __init__(self, DB, DB_Test):
        self.W = {}
        
        ETot = 0
        NTot = 0
        for ID in DB_Test:
            ERef = np.abs(DB.RefEnergies(ID))
            self.W[ID] = ERef.mean()
            
            ETot += np.sum(ERef)
            NTot += len(ERef)
        EMean = ETot/NTot
        WMean = np.mean([self.W[ID] for ID in DB_Test])
        
        for ID in DB_Test:
            self.W[ID] *= EMean/WMean
            
    def ErrFn(self, E, EB, ID):
        return self.W[ID]*(E-EB)

# Nice array
def NiceArr(X, pre='[ ', post=' ]', NPage=10):
    if len(X)>NPage:
        F = ""
        for k0 in range(0, len(X), NPage):
            k1 = min(len(X), k0 + NPage)
            if k0==0: pre, post = '[ ', '  '
            elif k1==len(X): pre, post = '  ', ' ]'
            else: pre, post = '  ', '  '
            F += NiceArr(X[k0:k1], pre, post, NPage)
        return F
    else:
        return pre + ", ".join(["%4.1f"%(x) for x in X]) + post + "\n"

# Nice dictionary
def NiceDict(X):
    R = []
    for v in list(X):
        R += ["%s: %.3f"%(v, X[v])]
    return ", ".join(R)

# Paginate a list
def Paginate(X, Line=72):
    Lines = []
    L = ""
    for x in X:
        N = " "+x
        if len(L) + len(N)>Line:
            Lines += [L]
            L = N
        else:
            L += N

    Lines += [L]
    return "\n".join(Lines)

# Percent ticker (because I'm impatient)
class TickOver:
    def __init__(self, N, Show=5):
        self.N = N
        self.Show = int(Show)
        self.LastTick = -10

    def Set(self, i):
        f = int(np.floor(i/self.N * 100./self.Show))
        if f>=(self.LastTick + 1):
            self.LastTick = f
            p = int(np.round(i/self.N * 100.))
            print("Done %3d%% - %5d of %5d"%(p, i, self.N))

# This has the routines for specialised subsets
class ExtinctionHelper:
    def __init__(self, DB, DB_Test,
                 NParam=2, DFAProps=None,
                 AllKey='All',
                 ErrMode = 'T'):
        # Initialise the database and other properties
        self.DB = DB
        self.AllKey = AllKey
        
        self.DB_Test = DB_Test
        self.ErrOpt = {}

        self.ErrMode = ErrMode[0].upper()
        if self.ErrMode=='E':
            self.ErrFn = EnergyErrFn
        elif self.ErrMode=='W':
            self.WTMAD2 = WTMAD2(DB, DB_Test)
            self.ErrFn = self.WTMAD2.ErrFn
        else:
            self.ErrMode = 'T'
            self.ErrFn = DefaultErrFn
        
        self.Names = DB.FullCache['Reactions'][self.AllKey]
        self.NAll = len(self.Names)

        # Initialise the DFA stuff
        self.InitDFA(NParam=NParam, DFAProps=DFAProps)

        # Defaults to False
        self.Quick = False
        # Defaults to MAE
        self.pError = 1 # Sum of |MAE|^pError

    def InitDFA(self, NParam=1, DFAProps=None):
        self.NParam = NParam
        self.DFAProps = DFAProps # Controls other properties (e.g. 'BLYP')
        if self.NParam==-1:
            # Setting NParam to -1 does PBE0
            self.OptFn = self.DB.OptPBE0
            self.DFAFn = DFA_PBE0
        else:
            # Otherwise does an XYG family with details (e.g. BLYP)
            # controlled by DFAProps
            self.OptFn = self.DB.OptXYG

            # Make sure the DFAFn matches
            if DFAProps is None: self.DFAFn = DFA_XYG
            elif str(DFAProps).upper()=='BLYP': self.DFAFn = DFA_XYG_BLYP
            elif str(DFAProps).upper()=='BPBE': self.DFAFn = DFA_XYG_BPBE
            else: self.DFAFn = DFA_XYG
            
        for ID in self.DB_Test:
            self.DB.Update(ID)
            c, self.ErrOpt[ID] \
                = self.OptFn(ID, NParam=self.NParam,
                             DFAProps=self.DFAProps,
                             RMSD=True)


    def SetQuick(self, Val=True): # Run the quicker version
        self.Quick = Val
        
    def SetPowerError(self, p):  # Sum of |MAE|^pError
        self.pError = p
            
    def TestTransferability(self, T=None):
        NTest = len(self.DB_Test)
        Err = { }
        MaxErr = { }
        for ID1 in self.DB_Test + ['All']:
            Err[ID1] = {}
            MaxErr[ID1] = 0.
            c, _ = self.OptFn(ID1, NParam=self.NParam,
                              DFAProps=self.DFAProps,
                              RMSD=True)
            DFAOpt = self.DFAFn(c)
            for ID2 in self.DB_Test:
                Err[ID1][ID2] = self.ErrFn(self.DB.GetMAD(ID2, DFAOpt),
                                           self.ErrOpt[ID2], ID2)
                if Err[ID1][ID2]>MaxErr[ID1]:
                    MaxErr[ID1] = Err[ID1][ID2]

        # Sort the errors from largest to smallest
        MaxErr = {k: v for k, v in
                  sorted(MaxErr.items(), key=lambda item: -item[1])}

        # Test on transferable pairs
        if not(T is None):
            for K1 in range(NTest):
                for K2 in range(K1+1,NTest):
                    ID1 = self.DB_Test[K1]
                    ID2 = self.DB_Test[K2]
                    if max(Err[ID1][ID2], Err[ID2][ID1])<1+T:
                        print("Sets %s and %s are similar - "%(ID1, ID2)\
                              + "Errors are %5.2f and %5.2f"\
                              %(Err[ID1][ID2], Err[ID2][ID1]))

        return Err, MaxErr
    
    def ComputeError(self, kk):
        kk = np.array(kk, dtype=int)
                  
        c, _ = self.OptFn(self.AllKey, kk=kk, NParam=self.NParam,
                          DFAProps=self.DFAProps,
                          RMSD=self.Quick)
        DFAOpt = self.DFAFn(c)

        Err = 0.
        for ID in self.DB_Test:
            Err += self.ErrFn(self.DB.GetMAD(ID, DFAOpt),
                              self.ErrOpt[ID], ID) ** self.pError
        
        return DFAOpt, Err/len(self.DB_Test)
        

    def Prune(self, kk, Err, NSurvive, ComputeMean=False):
        if ComputeMean:
            self.ErrMean = Err.mean()
            self.Err = Err*1.
            
        # The best survive
        kSurvive = np.argsort(Err)[:NSurvive]
        kkSurvive =kk[kSurvive,:]
        ErrSurvive = Err[kSurvive]

        NormErr = ErrSurvive/self.ErrMean

        print("Best = %4.1f%%, Worst = %4.1f%% relative to pure chance"\
              %(NormErr.min()*100, NormErr.max()*100))

        return kkSurvive, ErrSurvive

    def ReportBest(self, kkBest=None, kk=None, Err=None,
                   ShowAll=True):
        if kkBest is None:
            kBest = np.argmin(Err)
            kkBest = kk[kBest,:]
        DFABest, Err_Best = self.ComputeError(kkBest)

        self.Err_Best = Err_Best
        self.kkBest = kkBest
        self.DFABest = DFABest

        NTest = len(self.DB_Test)
        Err = {}
        for ID1 in self.DB_Test:
            Err[ID1] = self.ErrFn(self.DB.GetMAD(ID1, DFABest),
                                  self.ErrOpt[ID1], ID1)



        #print("#"*72)
        print("Best possible error = %8.5f [%6.1f%% of pure chance]"\
              %(Err_Best, Err_Best/self.ErrMean*100))
        
        if ShowAll:
            print("Optimal set is:")
            print(Paginate(['[ ']
                           + sorted(["\'%s\',"%(self.Names[k]) for k in kkBest])
                           + [' ]']))

            print("#"*72)
            print("Optimal DFA is:")
            DFAArr = ["{ ",]
            for Q in DFABest:
                if np.abs(DFABest[Q])>0.0005:
                    DFAArr += ["\'%s\': %6.3f,"%(Q, DFABest[Q])]
            DFAArr += [" }"]
            print(Paginate(DFAArr))

            print("#"*72)
            print("Transferabilities:")
            TrArr = ["{ ",]
            for ID in Err:
                TrArr += ["\'%s\': %6.3f,"%(ID, Err[ID])]
            TrArr += [" }"]
            print(Paginate(TrArr))

        return Err, DFABest, kkBest

    # Do the breeding
    def BreedSet(self, NSample=100, NSurvive=50,
                 NBacterial=100, NBreed=100,
                 UseAlpha=True, PeakGenePool=0.1,
                 UseCache=False,
                 Suff=None,
                 ):

        if Suff is None:
            Suff = "_NoSuffixSpecified"

        
        # Calculate the bacteria (random sampling with no breeding)
        Err = np.zeros((NBacterial,))
        kk  = np.zeros((NBacterial, NSample), dtype=int)

        # Use a cache if asked and possible
        NCached = 0
        RCacheFile = "Cache/_CacheBreeding_%s.npz"%(Suff)
        if UseCache:
            try:
                X = np.load(RCacheFile)
                Err_ = X['Err']
                kk_ = X['kk']
                NCached = X['NCached']

                # Reset if different number samples as is useless
                if not(kk_.shape[1]==NSample): 
                    NCached = 0
                elif NCached<NBacterial:
                    Err[:NCached] = Err_
                    kk[:NCached,:] = kk_
                elif NCached>=NBacterial:
                    Err = Err_[:NBacterial]
                    kk =kk_[:NBacterial]
                    NCached = NBacterial
            except:
                Err = np.zeros((NBacterial,))
                kk  = np.zeros((NBacterial, NSample), dtype=int)
                NCached = 0

        TO = TickOver(NBacterial) # This is to let us know what is going on
        for Iteration in range(NCached, NBacterial):
            # Pick a truly random sample
            kk[Iteration,:] = ra.choice(range(self.NAll), NSample,
                                        replace=False)
            # Compute the error
            _, Err[Iteration] = self.ComputeError(kk[Iteration,:])

            TO.Set(Iteration)

            # Update the cache every so often
            if (Iteration%50)==0:
                np.savez(RCacheFile,
                         Err=Err, kk=kk, NCached=Iteration)

        # Update the cache on completion
        np.savez(RCacheFile,
                 Err=Err, kk=kk, NCached=NBacterial)

        # Prune the full list so only the best survive
        kkSurvive, ErrSurvive = self.Prune(kk, Err, NSurvive,
                                           ComputeMean=True)

        # Now we run the breeding stage
        NSuccess = 0
        NBred = 0
        TO = TickOver(NBreed) # This is to let us know what is going on
        for Iteration in range(NBreed):
            # Pick a pair to breed
            if not(UseAlpha): # No alpha so both random
                Parent1, Parent2 = tuple(ra.choice(range(NSurvive),2,
                                                   replace=False))
            else: # Pick the alpha (best) and another random
                q = np.argsort(ErrSurvive)
                Parent1 = q[0]
                Parent2 = q[0]
                while Parent2==Parent1:
                    Parent2 = ra.randint(NSurvive)

            # Get the genes of both parents
            kk1 = set(kkSurvive[Parent1,:])
            kk2 = set(kkSurvive[Parent2,:])

            # Keep any shared genes
            kkShared = kk1 & kk2
            if len(kkShared)>0:
                # Randomly select from the remaining genes
                kkSeparate = ((kk1 | kk2) - kkShared)
                NRandom = NSample - len(kkShared)
                # Join the shared and random together
                kkN = np.hstack((list(kkShared),
                                 ra.choice(list(kkSeparate), NRandom,
                                           replace=False)))
            else:
                # No shared elements = random genes from both
                kkN = ra.choice(list(kk1)+list(kk2), NSample,
                                replace=False)

            # Calculate the error for the child
            _, ErrN = self.ComputeError(kkN)


            # Add the child to the gene pool by replacing the weakest
            if ErrN<ErrSurvive.max():
                kWorst = np.argmax(ErrSurvive)
                kkSurvive[kWorst,:] = kkN
                ErrSurvive[kWorst] = ErrN

                NSuccess += 1

            if (ErrSurvive.max()-ErrSurvive.min())<PeakGenePool/100.:
                print("Gene pool is no longer improving")
                break

            NBred += 1

            TO.Set(Iteration)

        print("Percent successful = %5.1f%%"%(NSuccess/NBred*100.))

        kkSurvive, ErrSurvive = self.Prune(kkSurvive, ErrSurvive, NSurvive)

        # And the very best is...
        Err_Best, DFABest, kkBest \
            = self.ReportBest(kk=kkSurvive, Err=ErrSurvive)

        return Err_Best, DFABest, kkBest
        

    # Read and write a file that contains all results
    # ReadSaveAllSet(Suff, Run=[I]) gives Run I from the file
    # ReadSaveAllSet(Suff, kkBest=...) adds kkBest to the file
    # ReadSaveAllSet(Suff, kkBest=..., Reset=True) overwrites the
    #     file with just kkBest (i.e. blanks earlier runs)
    def ReadSaveAllSet(self, Suff, Run=-1, kkBest=None,
                       Reset=False):
        # First load what is there
        CacheFile = "BenchmarkSets/AllTransferable_%s.npz"%(Suff)
        try:
            X = np.load(CacheFile)
            kkAll = np.array(X['kkAll'], dtype=int)
            ErrMeanAll = X['ErrMeanAll']
        except:
            kkAll = None
            ErrMeanAll = None
    
        if not(kkBest is None):
            if kkAll is None or Reset:
                kkAll = kkBest.reshape((-1,1))
                ErrMeanAll = np.array([self.ErrMean])
            else:
                kkAll = np.hstack((kkAll, kkBest.reshape((-1,1))))
                ErrMeanAll = np.hstack((ErrMeanAll, self.ErrMean))
            np.savez("BenchmarkSets/AllTransferable_%s.npz"%(Suff),
                     kkAll=kkAll, ErrMeanAll=ErrMeanAll)

            return kkBest, self.ErrMean
        else:
            if kkAll is None:
                print("Failed on read of %s quitting"%(CacheFile))
                quit()

            NRun = kkAll.shape[1]
            MRun = (Run+NRun) % NRun
            print("Reading Run %d in [0,%d]"\
                  %(MRun, NRun-1))
            return kkAll[:,Run], ErrMeanAll[Run]

    def SaveSet(self, Suff, kkBest=None, TxtOnly=False):
        if kkBest is None: kkBest=self.kkBest

        if not(TxtOnly):
            self.ReadSaveAllSet(Suff, kkBest=kkBest)
        
        F = open("BenchmarkSets/Transferable_%s.txt"%(Suff), "w")
        for k in self.kkBest:
            F.write(self.Names[k] + " # %d\n"%(k))
        F.close()

    def ReadSet(self, Suff, Run=-1):
        kkBest, ErrMean = self.ReadSaveAllSet(Suff, Run=Run)
        
        self.ErrMean = ErrMean
        self.kkBest = np.array(kkBest, dtype=int)
        return self.kkBest

    def NInSet(self, Suff):
        # First load what is there
        CacheFile = "BenchmarkSets/AllTransferable_%s.npz"%(Suff)
        try:
            X = np.load(CacheFile)
            ErrMeanAll = X['ErrMeanAll']
            return len(ErrMeanAll)
        except:
            return 0

