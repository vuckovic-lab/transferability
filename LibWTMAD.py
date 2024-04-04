from LibDB import *

W1_GMTKN55 = ['ACONF', 'ADIM6', 'AHB21', 'AL2X6', 'ALK8', 'ALKBDE10',
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


class DFADB_WTMAD(DFADB):
    def ComputeWTMAD(self, ID, Kind=2):
        # Create the weights by set
        ESet = {}
        for R, E in zip(
                self.FullCache['Reactions'][ID],
                self.FullCache['RefEnergies'][ID]
                ):
            Set = R.split(':')[0]

            if not(Set in ESet): ESet[Set] = [E,]
            else: ESet[Set] += [E,]

        WSet = { Set: 1. for Set in ESet }
        if Kind==2:
            EM = {}
            for Set in ESet:
                EM[Set] = np.mean(np.abs(ESet[Set]))

            EMAll = np.mean([EM[Set] for Set in EM])
            WSet = { Set: EMAll/EM[Set] for Set in EM }

        # Then, make them match the
        if not('Weights' in self.FullCache):
            self.FullCache['Weights'] = {}

        self.FullCache['Weights'][ID] = []
        for R, E in zip(
                self.FullCache['Reactions'][ID],
                self.FullCache['RefEnergies'][ID]
                ):
            Set = R.split(':')[0]
            self.FullCache['Weights'][ID] += [WSet[Set]]

        # Turn it into an array
        self.FullCache['Weights'][ID] \
            = np.array(self.FullCache['Weights'][ID])

    # Get the errors
    def GetErrors(self, ID, DFA, WithMF=True, kk=None):
        if kk is None:
            ERef = self.FullCache['RefEnergies'][ID]
            WRef = self.FullCache['Weights'][ID]
        else:
            ERef = self.FullCache['RefEnergies'][ID][kk]
            WRef = self.FullCache['Weights'][ID][kk]
        return WRef * (
            self.GetEnergies(ID, DFA,
                             WithMF=WithMF, kk=kk)
            - ERef )

    # Get the MAD on the set
    def GetMAD(self, ID, DFA, WithMF=True, kk=None):
        return np.mean(np.abs(self.GetErrors(ID, DFA,
                                             WithMF=WithMF, kk=kk)))

if __name__ == "__main__":
    DB = DFADB_WTMAD()
    DB.Combine('GMTKN55', DB_GMTKN55)
    DB.ComputeWTMAD('GMTKN55', Kind=2)
    print(DB.GetMAD('GMTKN55', DFA_PBE0(0.)))
