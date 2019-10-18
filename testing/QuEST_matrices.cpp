/** @file
 * Simple implementations of matrix operations used by QuEST_unit_tests
 *
 * @author Tyson Jones
 */

#include "QuEST.h"
#include "QuEST_complex.h"
#include "QuEST_matrices.hpp"
#include "catch.hpp"

// matrix ops I need:
// [X] complex matrices
// [X] complex vectors
// [X] multiply square matrices
// [X] populate square sub-matrices from other matrices
// [X] tensor product matrices
// [X] generate identity matrices
// [X] conjugate-transpose transpose matrices 
// [X] generate SWAP gate
// [X] multiply matrix onto vector
// [ ] compare vectors
// [ ] compare matrices

// unitary ops I need:
// [X] convert {ctrls, targs, matr, numQubits} to full-state matrix

// internal ops:
// [ ] convert/load statevec to complex vector 
// [ ] convert/load density matrix to complex matrix

/** produces a dim-by-dim square complex matrix, initialised to zero 
 */
QMatrix getZeroMatrix(size_t dim) {
    REQUIRE( dim > 1 );
    QMatrix matr = QMatrix(dim);
    for (size_t i=0; i<dim; i++)
        matr[i].resize(dim);
    return matr;
}

/** produces a dim-by-dim identity matrix 
 */
QMatrix getIdentityMatrix(size_t dim) {
    REQUIRE( dim > 1 );
    QMatrix matr = getZeroMatrix(dim);
    for (size_t i=0; i<dim; i++)
        matr[i][i] = 1;
    return matr;
}

/** returns a (otimes) b, where a and b are square but possibly different-sized 
 */
QMatrix getKroneckerProduct(QMatrix a, QMatrix b) {
    QMatrix prod = getZeroMatrix(a.size() * b.size());
    for (size_t r=0; r<b.size(); r++)
        for (size_t c=0; c<b.size(); c++)
            for (size_t i=0; i<a.size(); i++)
                for (size_t j=0; j<a.size(); j++)
                    prod[r+b.size()*i][c+b.size()*j] = a[i][j] * b[r][c];
    return prod;
}

/** returns a square matrix, the product of a and b 
 */
QMatrix getMatrixProduct(QMatrix a, QMatrix b) {
    REQUIRE( a.size() == b.size() );
    QMatrix prod = getZeroMatrix(a.size());
    for (size_t r=0; r<a.size(); r++)
        for (size_t c=0; c<a.size(); c++)
            for (size_t k=0; k<a.size(); k++)
                prod[r][c] += a[r][k] * b[k][c];
    return prod;
}

/** returns the conjugate transpose of the complex square matrix a 
 */
QMatrix getConjugateTranspose(QMatrix a) {
    QMatrix b = a;
    for (size_t r=0; r<a.size(); r++)
        for (size_t c=0; c<a.size(); c++)
            b[r][c] = conj(a[c][r]);
    return b;
}

/** modifies dest by overwriting its submatrix (from top-left corner 
 * (r, c) to bottom-right corner (r+dest.size(), c+dest.size()) with the 
 * complete elements of sub 
 */
void setSubMatrix(QMatrix &dest, QMatrix sub, size_t r, size_t c) {
    REQUIRE( sub.size() + r <= dest.size() );
    REQUIRE( sub.size() + c <= dest.size() );
    for (size_t i=0; i<sub.size(); i++)
        for (size_t j=0; j<sub.size(); j++)
            dest[r+i][c+j] = sub[i][j];
}

/** returns the 2^numQb-by-2^numQb unitary matrix which swaps qubits qb1 and qb2.
 * If qb1==qb2, returns the identity matrix.
 */
QMatrix getSwapMatrix(int qb1, int qb2, int numQb) {
    REQUIRE( numQb > 1 );
    REQUIRE( (qb1 >= 0 && qb1 < numQb) );
    REQUIRE( (qb2 >= 0 && qb2 < numQb) );
    
    if (qb1 > qb2)
        std::swap(qb1, qb2);
        
    if (qb1 == qb2)
        return getIdentityMatrix(1 << numQb);

    QMatrix swap;
    
    if (qb2 == qb1 + 1) {
        // qubits are adjacent
        swap = QMatrix{{1,0,0,0},{0,0,1,0},{0,1,0,0},{0,0,0,1}};
        
    } else {
        // qubits are distant
        int block = 1 << (qb2 - qb1);
        swap = getZeroMatrix(block*2);
        QMatrix iden = getIdentityMatrix(block/2);
        
        // Lemma 3.1 of arxiv.org/pdf/1711.09765.pdf
        QMatrix p0{{1,0},{0,0}};
        QMatrix l0{{0,1},{0,0}};
        QMatrix l1{{0,0},{1,0}};
        QMatrix p1{{0,0},{0,1}};
        
        /* notating a^(n+1) = identity(1<<n) (otimes) a, we construct the matrix
         * [ p0^(N) l1^N ]
         * [ l0^(N) p1^N ]
         * where N = qb2 - qb1 */
        setSubMatrix(swap, getKroneckerProduct(iden, p0), 0, 0);
        setSubMatrix(swap, getKroneckerProduct(iden, l0), block, 0);
        setSubMatrix(swap, getKroneckerProduct(iden, l1), 0, block);
        setSubMatrix(swap, getKroneckerProduct(iden, p1), block, block);
    }
    
    // pad swap with outer identities
    if (qb1 > 0)
        swap = getKroneckerProduct(getIdentityMatrix(1<<qb1), swap);
    if (qb2 < numQb-1)
        swap = getKroneckerProduct(swap, getIdentityMatrix(1<<(numQb-qb2-1)));
        
    return swap;
}

/** iterates list1 (of length len1) and replaces element oldEl with newEl, which is 
 * gauranteed to be present at most once (between list1 AND list2), though may 
 * not be present at all. If oldEl isn't present in list1, does the same for list2. 
 * list1 is skipped if == NULL. This is used by getFullOperatorMatrix() to ensure
 * that, when qubits are swapped, their appearences in the remaining qubit lists 
 * are updated.
 */
void updateIndices(int oldEl, int newEl, int* list1, int len1, int* list2, int len2) {
    if (list1 != NULL) {
        for (int i=0; i<len1; i++) {
            if (list1[i] == oldEl) {
                list1[i] = newEl;
                return;
            }
        }
    }
    for (int i=0; i<len2; i++) {
        if (list2[i] == oldEl) {
            list2[i] = newEl;
            return;
        }
    }
}

/** takes a 2^numTargs-by-2^numTargs matrix op and a returns a 2^numQubits-by-2^numQubits
 * matrix where op is controlled on the given ctrls qubits. The union of {ctrls}
 * and {targs} must be unique, and every element must be 0 or positive. 
 * The passed {ctrls} and {targs} arrays are unmodified.
 * This funciton works by first swapping {ctrls} and {targs} (via swap unitaries) 
 * to be strictly increasing {0,1,...}, building controlled(op), tensoring it to 
 * the full Hilbert space, and then 'unswapping'. The returned matrix has form:
 * swap1 ... swapN . c(op) . swapN ... swap1
 */
QMatrix getFullOperatorMatrix(
    int* ctrls, int numCtrls, int *targs, int numTargs, QMatrix op, int numQubits)
{        
    REQUIRE( numCtrls >= 0 );
    REQUIRE( numTargs >= 0 );
    REQUIRE( numQubits >= (numCtrls+numTargs) );
    REQUIRE( op.size() == (1 << numTargs) );
    
    // copy {ctrls} and {targs}to restore at end
    std::vector<int> ctrlsCopy(ctrls, ctrls+numCtrls);
    std::vector<int> targsCopy(targs, targs+numTargs);
    
    // full-state matrix of qubit swaps
    QMatrix swaps = getIdentityMatrix(1 << numQubits);
    QMatrix unswaps = getIdentityMatrix(1 << numQubits);
    QMatrix matr;
    
    // swap ctrls to {0, ..., numCtrls-1}
    for (int i=0; i<numCtrls; i++) {
        if (i != ctrls[i]) {
            matr = getSwapMatrix(i, ctrls[i], numQubits);
            swaps = getMatrixProduct(matr, swaps);
            unswaps = getMatrixProduct(unswaps, matr);
            
            // even if this is the last ctrl, targs might still need updating
            updateIndices(
                i, ctrls[i], (i < numCtrls-1)? &ctrls[i+1] : NULL, 
                numCtrls-i-1, targs, numTargs);
        }
    }

    // swap targs to {numCtrls, ..., numCtrls+numTargs-1}
    for (int i=0; i<numTargs; i++) {
        int newInd = numCtrls+i;
        if (newInd != targs[i]) {
            matr = getSwapMatrix(newInd, targs[i], numQubits);
            swaps = getMatrixProduct(matr, swaps);
            unswaps = getMatrixProduct(unswaps, matr);
            
            // update remaining targs (if any exist)
            if (i < numTargs-1)
                updateIndices(newInd, targs[i], NULL, 0, &targs[i+1], numTargs-i-1);
        }
    }
    
    // construct controlled-op matrix for qubits {0, ..., numCtrls+numTargs-1}
    size_t dim = 1 << (numCtrls+numTargs);
    QMatrix fullOp = getIdentityMatrix(dim);
    setSubMatrix(fullOp, op, dim-op.size(), dim-op.size());
    
    // create full-state controlled-op matrix (left-pad identities)
    if (numQubits > numCtrls+numTargs) {
        size_t pad = 1 << (numQubits - numCtrls - numTargs);
        fullOp = getKroneckerProduct(getIdentityMatrix(pad), fullOp);
    }
    
    // apply swap to either side (to swap qubits back and forth)
    fullOp = getMatrixProduct(fullOp, swaps);
    fullOp = getMatrixProduct(unswaps, fullOp);
    
    // restore {ctrls and targs}
    for (int i=0; i<numCtrls; i++)
        ctrls[i] = ctrlsCopy[i];
    for (int i=0; i<numTargs; i++)
        targs[i] = targsCopy[i];

    return fullOp;
}

/** returns the product of complex matrix m onto complex vector v 
 */
QVector getMatrixVectorProduct(QMatrix m, QVector v) {
    REQUIRE( m.size() == v.size() );
    QVector prod = QVector(v.size());
    for (size_t r=0; r<v.size(); r++)
        for (size_t c=0; c<v.size(); c++)
            prod[r] += m[r][c] * v[c];
    return prod;
}

#define macro_copyComplexMatrix(dest, src) { \
    for (size_t i=0; i<dest.size(); i++) \
        for (size_t j=0; j<dest.size(); j++) \
            dest[i][j] = qcomp(src.real[i][j], src.imag[i][j]); \
}
QMatrix toMatrix(ComplexMatrix2 src) {
    QMatrix dest = getZeroMatrix(2);
    macro_copyComplexMatrix(dest, src);
    return dest;
}
QMatrix toMatrix(ComplexMatrix4 src) {
    QMatrix dest = getZeroMatrix(4);
    macro_copyComplexMatrix(dest, src);
    return dest;
}
QMatrix toMatrix(ComplexMatrixN src) {
    QMatrix dest = getZeroMatrix(1 << src.numQubits);
    macro_copyComplexMatrix(dest, src);
    return dest;
}

