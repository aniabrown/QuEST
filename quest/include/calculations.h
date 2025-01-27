/** @file
 * API signatures for calculating properties of quantum states,
 * such as probabilities and expectation values.
 */

#ifndef CALCULATIONS_H
#define CALCULATIONS_H

#include "quest/include/types.h"
#include "quest/include/qureg.h"
#include "quest/include/paulis.h"
#include "quest/include/matrices.h"


// enable invocation by both C and C++ binaries
#ifdef __cplusplus
extern "C" {
#endif


qreal calcExpecPauliStr(Qureg qureg, PauliStr str);

qreal calcExpecPauliStrSum(Qureg qureg, PauliStrSum sum);

qreal calcExpecFullStateDiagMatr(Qureg qureg, FullStateDiagMatr matr);

qreal calcExpecFullStateDiagMatrPower(Qureg qureg, FullStateDiagMatr matr, qcomp exponent);


qreal calcTotalProb(Qureg qureg);

qreal calcProbOfBasisState(Qureg qureg, qindex index);

qreal calcProbOfQubitOutcome(Qureg qureg, int qubit, int outcome);

qreal calcProbOfMultiQubitOutcome(Qureg qureg, int* qubits, int* outcomes, int numQubits);

void  calcProbsOfAllMultiQubitOutcomes(qreal* outcomeProbs, Qureg qureg, int* qubits, int numQubits);


qreal calcPurity(Qureg qureg);

qreal calcFidelity(Qureg qureg, Qureg other);

qreal calcDistance(Qureg qureg1, Qureg qureg2);


// end de-mangler
#ifdef __cplusplus
}
#endif



/*
 * C++ ONLY FUNCTIONS
 *
 * which are not directly C-compatible because they pass or
 * return qcomp primitives by-value (rather than by pointer).
 * This is prohibited because the C and C++ ABI does not agree
 * on a complex type, though C's _Complex has the same memory
 * layout as C++'s std::complex<>. To work around this, the 
 * below functions have a C-compatible wrapper defined in
 * wrappers.h which passes/receives the primitives by pointer;
 * a qcomp ptr can be safely passed from the C++ source binary
 * the user's C binary.
 */

#ifdef __cplusplus

qcomp calcInnerProduct(Qureg qureg1, Qureg qureg2);

qcomp calcExpecNonHermitianPauliStrSum(Qureg qureg, PauliStrSum sum); 

qcomp calcExpecNonHermitianFullStateDiagMatr(Qureg qureg, FullStateDiagMatr matr);

qcomp calcExpecNonHermitianFullStateDiagMatrPower(Qureg qureg, FullStateDiagMatr matrix, qcomp exponent);

#endif



#endif // CALCULATIONS_H