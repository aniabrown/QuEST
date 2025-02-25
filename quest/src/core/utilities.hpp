/** @file
 * Miscellaneous utility functions needed internally.
 */

#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include "quest/include/types.h"
#include "quest/include/qureg.h"
#include "quest/include/paulis.h"
#include "quest/include/matrices.h"
#include "quest/include/channels.h"
#include "quest/include/environment.h"

#include <type_traits>
#include <string>
#include <vector>
#include <array>

using std::is_same_v;
using std::vector;



/*
 * QUBIT PROCESSING
 */

bool util_isQubitInSuffix(int qubit, Qureg qureg);
bool util_isBraQubitInSuffix(int ketQubit, Qureg qureg);

int util_getBraQubit(int ketQubit, Qureg qureg);

int util_getPrefixInd(int qubit, Qureg qureg);
int util_getPrefixBraInd(int ketQubit, Qureg qureg);

std::array<vector<int>,2> util_getPrefixAndSuffixQubits(vector<int> qubits, Qureg qureg);

int util_getRankBitOfQubit(int ketQubit, Qureg qureg);
int util_getRankBitOfBraQubit(int ketQubit, Qureg qureg);

int util_getRankWithQubitFlipped(int ketQubit, Qureg qureg);
int util_getRankWithQubitsFlipped(vector<int> prefixQubits, Qureg qureg);

int util_getRankWithBraQubitFlipped(int ketQubit, Qureg qureg);
int util_getRankWithBraQubitsFlipped(vector<int> ketQubits, Qureg qureg);

vector<int> util_getBraQubits(vector<int> ketQubits, Qureg qureg);

vector<int> util_getVector(int* qubits, int numQubits);

vector<int> util_getConcatenated(vector<int> list1, vector<int> list2);

vector<int> util_getSorted(vector<int> list);
vector<int> util_getSorted(vector<int> ctrls, vector<int> targs);

qindex util_getBitMask(vector<int> qubits);
qindex util_getBitMask(vector<int> qubits, vector<int> states);
qindex util_getBitMask(vector<int> ctrls, vector<int> ctrlStates, vector<int> targs, vector<int> targStates);



/*
 * INDEX ALGEBRA
 */

qindex util_getGlobalIndexOfFirstLocalAmp(Qureg qureg);
qindex util_getGlobalColumnOfFirstLocalAmp(Qureg qureg);

qindex util_getLocalIndexOfGlobalIndex(Qureg qureg, qindex globalInd);

qindex util_getLocalIndexOfFirstDiagonalAmp(Qureg qureg);

qindex util_getGlobalFlatIndex(Qureg qureg, qindex globalRow, qindex globalCol);

int util_getRankContainingIndex(Qureg qureg, qindex globalInd);
int util_getRankContainingColumn(Qureg qureg, qindex globalCol);
int util_getRankContainingIndex(FullStateDiagMatr matr, qindex globalInd);

qindex util_getNextPowerOf2(qindex number);



/*
 * COMPLEX ALGEBRA
 */

qcomp util_getPowerOfI(size_t exponent);



/*
 * STRUCT TYPING
 *
 * defined here in the header since templated, and which use compile-time inspection.
 */


constexpr bool util_isQuregType(auto&& x) { return is_same_v<std::decay_t<decltype(x)>, Qureg>; }
constexpr bool util_isCompMatr1(auto&& x) { return is_same_v<std::decay_t<decltype(x)>, CompMatr1>; }
constexpr bool util_isCompMatr2(auto&& x) { return is_same_v<std::decay_t<decltype(x)>, CompMatr2>; }
constexpr bool util_isCompMatr (auto&& x) { return is_same_v<std::decay_t<decltype(x)>, CompMatr >; }
constexpr bool util_isDiagMatr1(auto&& x) { return is_same_v<std::decay_t<decltype(x)>, DiagMatr1>; }
constexpr bool util_isDiagMatr2(auto&& x) { return is_same_v<std::decay_t<decltype(x)>, DiagMatr2>; }
constexpr bool util_isDiagMatr (auto&& x) { return is_same_v<std::decay_t<decltype(x)>, DiagMatr >; }
constexpr bool util_isFullStateDiagMatr(auto&& x) { return is_same_v<std::decay_t<decltype(x)>, FullStateDiagMatr>; }

constexpr bool util_isDenseMatrixType(auto&& x) {
    using T = std::decay_t<decltype(x)>;

    // CompMatr, SuperOp and (in a sense) KrausMaps are "dense", storing all 2D elements
    if constexpr (
        is_same_v<T, CompMatr1> ||
        is_same_v<T, CompMatr2> ||
        is_same_v<T, CompMatr>  ||
        is_same_v<T, KrausMap>  ||
        is_same_v<T, SuperOp>
    )
        return true;

    // DiagMatr are "sparse", storing only the diagonals
    if constexpr (
        is_same_v<T, DiagMatr1> ||
        is_same_v<T, DiagMatr2> ||
        is_same_v<T, DiagMatr>  ||
        is_same_v<T, FullStateDiagMatr>
    )
        return false;

    // this line is reached if the type is not a matrix
    return false;
}

constexpr bool util_isFixedSizeMatrixType(auto&& x) {
    using T = std::decay_t<decltype(x)>;

    return (
        is_same_v<T, CompMatr1> ||
        is_same_v<T, CompMatr2> ||
        is_same_v<T, DiagMatr1> ||
        is_same_v<T, DiagMatr2>
    );
}

constexpr bool util_isHeapMatrixType(auto&& x) {

    // all non-fixed size matrices are stored in the heap (never the stack)
    return ! util_isFixedSizeMatrixType(x);
}

constexpr bool util_isDistributableType(auto&& x) {
    using T = std::decay_t<decltype(x)>;
    return (is_same_v<T, FullStateDiagMatr> || is_same_v<T, Qureg>);
}

bool util_isDistributedMatrix(auto&& matr) {

    if constexpr (util_isDistributableType(matr))
        return matr.isDistributed;

    return false;
}

bool util_isGpuAcceleratedMatrix(auto&& matr) {

    if constexpr (util_isFullStateDiagMatr(matr))
        return matr.isGpuAccelerated;

    if constexpr (util_isHeapMatrixType(matr))
        return getQuESTEnv().isGpuAccelerated;

    return false;
}

std::string util_getMatrixTypeName(auto&& matr) {
    
    if constexpr (util_isCompMatr1(matr)) return "CompMatr1";
    if constexpr (util_isCompMatr2(matr)) return "CompMatr2";
    if constexpr (util_isCompMatr (matr)) return "CompMatr" ;
    if constexpr (util_isDiagMatr1(matr)) return "DiagMatr1";
    if constexpr (util_isDiagMatr2(matr)) return "DiagMatr2";
    if constexpr (util_isDiagMatr (matr)) return "DiagMatr" ;
    if constexpr (util_isFullStateDiagMatr(matr)) return "FullStateDiagMatr";

    // no need to create a new error for this situation
    return "UnrecognisedMatrix";
}

qindex util_getMatrixDim(auto&& matr) {
    
    if constexpr (util_isDenseMatrixType(matr))
        return matr.numRows;
    else
        return matr.numElems;
}

qcomp* util_getGpuMemPtr(auto&& matr) {

    // matr = CompMatr, DiagMatr, FullStateDiagMatr, 
    //        SuperOp, but NOT KrausMap

    // 2D CUDA structures are always stored as 1D
    if constexpr (util_isDenseMatrixType(matr))
        return matr.gpuElemsFlat;
    else
        return matr.gpuElems;
}



/*
 * MATRIX CONJUGATION
 */

CompMatr1 util_getConj(CompMatr1 matrix);
CompMatr2 util_getConj(CompMatr2 matrix);
DiagMatr1 util_getConj(DiagMatr1 matrix);
DiagMatr2 util_getConj(DiagMatr2 matrix);

void util_setConj(CompMatr matrix);
void util_setConj(DiagMatr matrix);



/*
 * MATRIX UNITARITY
 */

bool util_isUnitary(CompMatr1 matrix, qreal epsilon);
bool util_isUnitary(CompMatr2 matrix, qreal epsilon);
bool util_isUnitary(CompMatr  matrix, qreal epsilon);
bool util_isUnitary(DiagMatr1 matrix, qreal epsilon);
bool util_isUnitary(DiagMatr2 matrix, qreal epsilon);
bool util_isUnitary(DiagMatr  matrix, qreal epsilon);
bool util_isUnitary(FullStateDiagMatr matrix, qreal epsilon);



/*
 * MATRIX HERMITICITY
 */

bool util_isHermitian(CompMatr1 matrix, qreal epsilon);
bool util_isHermitian(CompMatr2 matrix, qreal epsilon);
bool util_isHermitian(CompMatr  matrix, qreal epsilon);
bool util_isHermitian(DiagMatr1 matrix, qreal epsilon);
bool util_isHermitian(DiagMatr2 matrix, qreal epsilon);
bool util_isHermitian(DiagMatr  matrix, qreal epsilon);
bool util_isHermitian(FullStateDiagMatr matrix, qreal epsilon);



/*
 * PAULI STR SUM HERMITICITY
 */

bool util_isHermitian(PauliStrSum sum, qreal epsilon);



/*
 * KRAUS MAPS AND SUPEROPERATORS
 */

bool util_isCPTP(KrausMap map, qreal epsilon);

void util_setSuperoperator(qcomp** superop, vector<vector<vector<qcomp>>> matrices, int numMatrices, int numQubits);
void util_setSuperoperator(qcomp** superop, qcomp*** matrices, int numMatrices, int numQubits);



/*
 * DISTRIBUTED ELEMENTS INDEXING
 */

struct util_VectorIndexRange {

    // the first local index of this node's amps which are in the queried distributed range
    qindex localDistribStartInd;

    // the corresponding local index of the non-distributed (i.e. duplicated on every node) data structure
    qindex localDuplicStartInd;

    // the number of this node's amps which are within the queried distributed range
    qindex numElems;
};

bool util_areAnyVectorElemsWithinNode(int rank, qindex numElemsPerNode, qindex startInd, qindex numInds);

util_VectorIndexRange util_getLocalIndRangeOfVectorElemsWithinNode(int rank, qindex numElemsPerNode, qindex elemStartInd, qindex numInds);



/*
 * GATE PARAMETERS
 */

qreal util_getPhaseFromGateAngle(qreal angle);



/*
 * DECOHERENCE FACTORS
 */

struct util_Scalars { qreal c1; qreal c2; qreal c3; qreal c4; };

qreal util_getOneQubitDephasingFactor(qreal prob);

qreal util_getTwoQubitDephasingTerm(qreal prob);

util_Scalars util_getOneQubitDepolarisingFactors(qreal prob);

util_Scalars util_getTwoQubitDepolarisingFactors(qreal prob);

util_Scalars util_getOneQubitDampingFactors(qreal prob);

util_Scalars util_getOneQubitPauliChannelFactors(qreal pI, qreal pX, qreal pY, qreal pZ);

qreal util_getMaxProbOfOneQubitDephasing();

qreal util_getMaxProbOfTwoQubitDephasing();

qreal util_getMaxProbOfOneQubitDepolarising();

qreal util_getMaxProbOfTwoQubitDepolarising();



/*
 * TEMPORARY MEMORY ALLOCATION
 */

void util_tryAllocVector(vector<qreal > &vec, qindex size, void (*errFunc)());
void util_tryAllocVector(vector<qcomp > &vec, qindex size, void (*errFunc)());
void util_tryAllocVector(vector<qcomp*> &vec, qindex size, void (*errFunc)());

// cuQuantum needs a vector<double> overload, which we additionally define when qreal!=double. Gross!
#if FLOAT_PRECISION != 2
void util_tryAllocVector(vector<double> &vec, qindex size, void (*errFunc)());
#endif

void util_tryAllocMatrix(vector<vector<qcomp>> &vec, qindex numRows, qindex numCols, void (*errFunc)());



#endif // UTILITIES_HPP