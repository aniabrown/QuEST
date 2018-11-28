// Distributed under MIT licence. See https://github.com/aniabrown/QuEST/blob/master/LICENCE.txt for details 

/** @file
 * An implementation of the pure backend in ../QuEST_ops_pure.h for a local (non-MPI, non-GPU) environment.
 * Mostly pure-state wrappers for the local/distributed functions implemented in QuEST_cpu
 */

# include "../QuEST.h"
# include "../QuEST_ops.h"
# include "../QuEST_internal.h"
# include "../QuEST_precision.h"
# include "../mt19937ar.h"

# include "QuEST_cpu_internal.h"

# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <sys/types.h>

# ifdef _OPENMP
# include <omp.h>
# endif


void densmatr_oneQubitDepolarise(QubitRegister qureg, const int targetQubit, REAL depolLevel) {
    if (depolLevel == 0)
        return;

    densmatr_oneQubitDepolariseLocal(qureg, targetQubit, depolLevel);
}

void densmatr_twoQubitDepolarise(QubitRegister qureg, int qubit1, int qubit2, REAL depolLevel){
    if (depolLevel == 0)
        return;
    REAL eta = 2/depolLevel;
    REAL delta = eta - 1 - sqrt( (eta-1)*(eta-1) - 1 );
    REAL gamma = 1+delta;
    // TODO -- test delta too small

    gamma = 1/(gamma*gamma*gamma);
    densmatr_twoQubitDephase(qureg, qubit1, qubit2, depolLevel);
    densmatr_twoQubitDepolariseLocal(qureg, qubit1, qubit2, delta, gamma);
}


REAL densmatr_calcPurity(QubitRegister qureg) {
    return densmatr_calcPurityLocal(qureg);
}

REAL densmatr_calcFidelity(QubitRegister qureg, QubitRegister pureState) {
    
    // save pointers to qureg's pair state
    REAL* quregPairRePtr = qureg.pairStateVec.real;
    REAL* quregPairImPtr = qureg.pairStateVec.imag;
    
    // populate qureg pair state with pure state (by repointing)
    qureg.pairStateVec.real = pureState.stateVec.real;
    qureg.pairStateVec.imag = pureState.stateVec.imag;
    
    // calculate fidelity using pairState
    REAL fid = densmatr_calcFidelityLocal(qureg, pureState);
    
    // restore pointers
    qureg.pairStateVec.real = quregPairRePtr;
    qureg.pairStateVec.imag = quregPairImPtr;
    
    return fid;
}

void densmatr_initPureState(QubitRegister qureg, QubitRegister pureState) {
    
    // save pointers to qureg's pair state
    REAL* quregPairRePtr = qureg.pairStateVec.real;
    REAL* quregPairImPtr = qureg.pairStateVec.imag;
    
    // populate qureg pair state with pure state (by repointing)
    qureg.pairStateVec.real = pureState.stateVec.real;
    qureg.pairStateVec.imag = pureState.stateVec.imag;

    // populate density matrix via it's pairState
    densmatr_initPureStateLocal(qureg, pureState);
    
    // restore pointers
    qureg.pairStateVec.real = quregPairRePtr;
    qureg.pairStateVec.imag = quregPairImPtr;
}

Complex statevec_calcInnerProduct(QubitRegister bra, QubitRegister ket) {
    return statevec_calcInnerProductLocal(bra, ket);
}

REAL densmatr_calcTotalProb(QubitRegister qureg) {
    
    // computes the trace using Kahan summation
    REAL pTotal=0;
    REAL y, t, c;
    c = 0;
    
    long long int numCols = 1LL << qureg.numQubitsRepresented;
    long long diagIndex;
    
    for (int col=0; col< numCols; col++) {
        diagIndex = col*(numCols + 1);
        y = qureg.stateVec.real[diagIndex] - c;
        t = pTotal + y;
        c = ( t - pTotal ) - y; // brackets are important
        pTotal = t;
    }
    
    // @TODO should maybe do a cheap test that imaginary components are ~0
    
    return pTotal;
}

REAL statevec_calcTotalProb(QubitRegister qureg){
    // implemented using Kahan summation for greater accuracy at a slight floating
    // point operation overhead. For more details see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    REAL pTotal=0; 
    REAL y, t, c;
    long long int index;
    long long int numAmpsPerRank = qureg.numAmpsPerChunk;
    c = 0.0;
    for (index=0; index<numAmpsPerRank; index++){ 
        // Perform pTotal+=qureg.stateVec.real[index]*qureg.stateVec.real[index]; by Kahan

        y = qureg.stateVec.real[index]*qureg.stateVec.real[index] - c;
        t = pTotal + y;
        // Don't change the bracketing on the following line
        c = ( t - pTotal ) - y;
        pTotal = t;

        // Perform pTotal+=qureg.stateVec.imag[index]*qureg.stateVec.imag[index]; by Kahan

        y = qureg.stateVec.imag[index]*qureg.stateVec.imag[index] - c;
        t = pTotal + y;
        // Don't change the bracketing on the following line
        c = ( t - pTotal ) - y;
        pTotal = t;


    } 
    return pTotal;
}


QuESTEnv createQuESTEnv(void) {
    // init MPI environment
    
    QuESTEnv env;
    env.rank=0;
    env.numRanks=1;
    
    seedQuESTDefault();
    
    return env;
}

void syncQuESTEnv(QuESTEnv env){
    // MPI Barrier goes here in MPI version. 
} 

int syncQuESTSuccess(int successCode){
    return successCode;
}

void destroyQuESTEnv(QuESTEnv env){
    // MPI finalize goes here in MPI version. Call this function anyway for consistency
}

void reportQuESTEnv(QuESTEnv env){
    printf("EXECUTION ENVIRONMENT:\n");
    printf("Running locally on one node\n");
    printf("Number of ranks is %d\n", env.numRanks);
# ifdef _OPENMP
    printf("OpenMP enabled\n");
    printf("Number of threads available is %d\n", omp_get_max_threads());
# else
    printf("OpenMP disabled\n");
# endif
    printf("Precision: size of REAL is %ld bytes\n", sizeof(REAL));
}

void reportNodeList(QuESTEnv env){
    printf("Hostname unknown: running locally\n");
}

REAL statevec_getRealAmpEl(QubitRegister qureg, long long int index){
    return qureg.stateVec.real[index];
}

REAL statevec_getImagAmpEl(QubitRegister qureg, long long int index){
    return qureg.stateVec.imag[index];
}

void statevec_compactUnitary(QubitRegister qureg, const int targetQubit, Complex alpha, Complex beta) 
{
    statevec_compactUnitaryLocal(qureg, targetQubit, alpha, beta);
}

void statevec_unitary(QubitRegister qureg, const int targetQubit, ComplexMatrix2 u) 
{
    statevec_unitaryLocal(qureg, targetQubit, u);
}

void statevec_controlledCompactUnitary(QubitRegister qureg, const int controlQubit, const int targetQubit, Complex alpha, Complex beta) 
{
    statevec_controlledCompactUnitaryLocal(qureg, controlQubit, targetQubit, alpha, beta);
}

void statevec_controlledUnitary(QubitRegister qureg, const int controlQubit, const int targetQubit, ComplexMatrix2 u) 
{
    statevec_controlledUnitaryLocal(qureg, controlQubit, targetQubit, u);
}

void statevec_multiControlledUnitary(QubitRegister qureg, int* controlQubits, const int numControlQubits, const int targetQubit, ComplexMatrix2 u) 
{
    long long int mask=0; 
    for (int i=0; i<numControlQubits; i++)
        mask = mask | (1LL<<controlQubits[i]);

    statevec_multiControlledUnitaryLocal(qureg, targetQubit, mask, u);
}

void statevec_pauliX(QubitRegister qureg, const int targetQubit) 
{
    statevec_pauliXLocal(qureg, targetQubit);
}

void statevec_pauliY(QubitRegister qureg, const int targetQubit) 
{
    int conjFac = 1;
    statevec_pauliYLocal(qureg, targetQubit, conjFac);
}

void statevec_pauliYConj(QubitRegister qureg, const int targetQubit) 
{
    int conjFac = -1;
    statevec_pauliYLocal(qureg, targetQubit, conjFac);
}

void statevec_controlledPauliY(QubitRegister qureg, const int controlQubit, const int targetQubit)
{
    int conjFac = 1;
    statevec_controlledPauliYLocal(qureg, controlQubit, targetQubit, conjFac);
}

void statevec_controlledPauliYConj(QubitRegister qureg, const int controlQubit, const int targetQubit)
{
    int conjFac = -1;
    statevec_controlledPauliYLocal(qureg, controlQubit, targetQubit, conjFac);
}

void statevec_hadamard(QubitRegister qureg, const int targetQubit) 
{
    statevec_hadamardLocal(qureg, targetQubit);
}

void statevec_controlledNot(QubitRegister qureg, const int controlQubit, const int targetQubit) 
{
    statevec_controlledNotLocal(qureg, controlQubit, targetQubit);
}

REAL statevec_calcProbOfOutcome(QubitRegister qureg, const int measureQubit, int outcome)
{
    REAL stateProb=0;
    stateProb = statevec_findProbabilityOfZeroLocal(qureg, measureQubit);
    if (outcome==1) stateProb = 1.0 - stateProb;
    return stateProb;
}

REAL densmatr_calcProbOfOutcome(QubitRegister qureg, const int measureQubit, int outcome) {
    
    REAL outcomeProb = densmatr_findProbabilityOfZeroLocal(qureg, measureQubit);
    if (outcome == 1)
        outcomeProb = 1.0 - outcomeProb;
    return outcomeProb;
}

void statevec_collapseToKnownProbOutcome(QubitRegister qureg, const int measureQubit, int outcome, REAL stateProb)
{
    statevec_collapseToKnownProbOutcomeLocal(qureg, measureQubit, outcome, stateProb);
}
