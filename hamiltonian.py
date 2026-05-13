from __future__ import annotations
from sympy import symbols, Matrix, Rational, sqrt, I, eye, pprint, Eq, latex, simplify, conjugate, linear_eq_to_matrix, N, solve, Add, solve_linear_system, re, im, julia_code
import scipy as sp
import copy
import logging
import numpy as np
from qpack import *


J11, J12, J13, J21, J22, J23, J31, J32, J33 = symbols('J11 J12 J13 J21 J22 J23 J31 J32 J33', real=True)
D11, D12, D13, D22, D23, D33 = symbols('D11 D12 D13 D22 D23 D33', real=True)
J = symbols('J', real=True)


def readeigvecs(fname, allowed_blks = {0: [0], 1: [0]}, maxstate=5):
    lindex = 0
    datalines = []
    veclist = []
    with open(fname, "r") as data:
        datalines = data.readlines()
    for i, l in enumerate(datalines):
        if "The threshold for printing is" in l:
            lindex = i
    tempstate = None
    for l in datalines[lindex+3:]:
        if "---------------------" in l: break
        if len(l.split())<1: continue
        if "STATE" in l:
            if tempstate != None:
                veclist.append(tempstate)
            tempstate = SumState.create_empty()
            linevals = l.strip().split()
            if int(linevals[1].strip(':')) > maxstate: break
            en = float(linevals[-1])
            tempstate.energy = en
            continue
        ldata = l.strip().split()
        if not int(ldata[4]) in allowed_blks: continue
        if not int(ldata[5]) in allowed_blks[int(ldata[4])]: continue
        a, b, s, ms = float(ldata[1]), float(ldata[2]), ldata[6], ldata[7]
        s = Rational(s) if not '/' in s else Rational(s.split('/')[0], s.split('/')[1])
        ms = Rational(ms) if not '/' in ms else Rational(ms.split('/')[0], ms.split('/')[1])
        tempstate.addstate(SumBasis(s, ms, coeff=a+b*I))
    return veclist

def readeigvecsmin(fname, allowed_blks = {0: [0]}, maxstate=3):
    lindex = 0
    datalines = []
    veclist = []
    with open(fname, "r") as data:
        datalines = data.readlines()
    for i, l in enumerate(datalines):
        if "The threshold for printing is" in l:
            lindex = i
    tempstate = None
    for l in datalines[lindex+3:]:
        if "---------------------" in l: break
        if len(l.split())<1: continue
        if "STATE" in l:
            if tempstate != None:
                veclist.append(tempstate)
            tempstate = SumState.create_empty()
            linevals = l.strip().split()
            if int(linevals[1].strip(':')) > maxstate: break
            en = float(linevals[-1])
            tempstate.energy = en
            continue
        ldata = l.strip().split()
        if not int(ldata[4]) in allowed_blks: continue
        if not int(ldata[5]) in allowed_blks[int(ldata[4])]: continue
        a, b, s, ms = float(ldata[1]), float(ldata[2]), ldata[6], ldata[7]
        s = Rational(s) if not '/' in s else Rational(s.split('/')[0], s.split('/')[1])
        ms = Rational(ms) if not '/' in ms else Rational(ms.split('/')[0], ms.split('/')[1])
        tempstate.addstate(SumBasis(s, ms, coeff=a+b*I))
    return veclist

def getnorm(eigstate):
    res = 0.0
    for s in eigstate.states:
        res += simplify(conjugate(s.coeff)*s.coeff)
    return sqrt(res)


def projectedH(states):
    eigstates = [SumState.create(Rational(3,2), Rational(3,2)),
                SumState.create(Rational(3,2), Rational(1,2)),
                SumState.create(Rational(3,2), Rational(-1,2)),
                SumState.create(Rational(3,2), Rational(-3,2)),
                SumState.create(Rational(1,2), Rational(1,2)),
                SumState.create(Rational(1,2), Rational(-1,2))]
    resmat = eye(len(eigstates), len(eigstates))
    for k,s1 in enumerate(eigstates):
        for l,s2 in enumerate(eigstates):
            term = SumState()
            for ps in states:
                ncoeff = ps.inner_product(s2)
                term = term.add(ps.scale(ncoeff*ps.energy))
            resmat[k,l] = simplify(s1.inner_product(term))
    return resmat

def projectedminH(states):
    eigstates = [SumState.create(Rational(3,2), Rational(3,2)),
                SumState.create(Rational(3,2), Rational(1,2)),
                SumState.create(Rational(3,2), Rational(-1,2)),
                SumState.create(Rational(3,2), Rational(-3,2))]
    resmat = eye(len(eigstates), len(eigstates))
    for k,s1 in enumerate(eigstates):
        for l,s2 in enumerate(eigstates):
            term = SumState()
            for ps in states:
                ncoeff = ps.inner_product(s2)
                term = term.add(ps.scale(ncoeff*ps.energy))
            resmat[k,l] = simplify(s1.inner_product(term))
    return resmat

def modelHaniso():
    eigstates = [ProductState.create(Rational(1,1), Rational(1,1), Rational(1,2), Rational(1,2)),
                ProductState.create(Rational(1,1), Rational(0,1), Rational(1,2), Rational(1,2), coeff=sqrt(6)/3).
                add(ProductState.create(Rational(1,1), Rational(1,1), Rational(1,2), Rational(-1,2), coeff=sqrt(3)/3)),
                ProductState.create(Rational(1,1), Rational(-1,1), Rational(1,2), Rational(1,2), coeff=sqrt(3)/3).
                add(ProductState.create(Rational(1,1), Rational(0,1), Rational(1,2), Rational(-1,2), coeff=sqrt(6)/3)),
                ProductState.create(Rational(1,1), Rational(-1,1), Rational(1,2), Rational(-1,2)),
                ProductState.create(Rational(1,1), Rational(0,1), Rational(1,2), Rational(1,2), coeff=-sqrt(3)/3).
                add(ProductState.create(Rational(1,1), Rational(1,1), Rational(1,2), Rational(-1,2), coeff=sqrt(6)/3)),
                ProductState.create(Rational(1,1), Rational(-1,1), Rational(1,2), Rational(1,2), coeff=-sqrt(6)/3).
                add(ProductState.create(Rational(1,1), Rational(0,1), Rational(1,2), Rational(-1,2), coeff=sqrt(3)/3))]
    spinops1 = [ProductOp.S1x, ProductOp.S1y, ProductOp.S1z]
    spinops2 = [ProductOp.S2x, ProductOp.S2y, ProductOp.S2z]
    dmat = Matrix([[D11, D12, D13],
                   [D12, D22, D23],
                   [D13, D23, D33]])
    exchtensor = Matrix([[J11, J12, J13],
                         [J21, J22, J23],
                         [J31, J32, J33]])
    resmat = eye(len(eigstates), len(eigstates))
    for k, s1 in enumerate(eigstates):
        for l, s2 in enumerate(eigstates):
            tempres = Rational(0)
            for i in range(3):
                for j in range(3):
                    tempres += dmat[i,j]*s1.inner_product(spinops1[i](spinops1[j](s2)))
                    tempres += exchtensor[i,j]*s1.inner_product(spinops1[i](spinops2[j](s2)))
            resmat[k,l] = tempres
    return resmat

def modelHzfs():
    eigstates = [ProductState.create(Rational(1,1), Rational(1,1), Rational(1,2), Rational(1,2)),
                ProductState.create(Rational(1,1), Rational(0,1), Rational(1,2), Rational(1,2), coeff=sqrt(6)/3).
                add(ProductState.create(Rational(1,1), Rational(1,1), Rational(1,2), Rational(-1,2), coeff=sqrt(3)/3)),
                ProductState.create(Rational(1,1), Rational(-1,1), Rational(1,2), Rational(1,2), coeff=sqrt(3)/3).
                add(ProductState.create(Rational(1,1), Rational(0,1), Rational(1,2), Rational(-1,2), coeff=sqrt(6)/3)),
                ProductState.create(Rational(1,1), Rational(-1,1), Rational(1,2), Rational(-1,2)),
                ProductState.create(Rational(1,1), Rational(0,1), Rational(1,2), Rational(1,2), coeff=-sqrt(3)/3).
                add(ProductState.create(Rational(1,1), Rational(1,1), Rational(1,2), Rational(-1,2), coeff=sqrt(6)/3)),
                ProductState.create(Rational(1,1), Rational(-1,1), Rational(1,2), Rational(1,2), coeff=-sqrt(6)/3).
                add(ProductState.create(Rational(1,1), Rational(0,1), Rational(1,2), Rational(-1,2), coeff=sqrt(3)/3))]
    spinops1 = [ProductOp.S1x, ProductOp.S1y, ProductOp.S1z]
    dmat = Matrix([[D11, D12, D13],
                   [D12, D22, D23],
                   [D13, D23, D33]])
    resmat = eye(len(eigstates), len(eigstates))
    for k, s1 in enumerate(eigstates):
        for l, s2 in enumerate(eigstates):
            tempres = Rational(0)
            for i in range(3):
                for j in range(3):
                    tempres += dmat[i,j]*s1.inner_product(spinops1[i](spinops1[j](s2)))
            resmat[k,l] = tempres
    return resmat

def modelHzfsJ():
    eigstates = [ProductState.create(Rational(1,1), Rational(1,1), Rational(1,2), Rational(1,2)),
                ProductState.create(Rational(1,1), Rational(0,1), Rational(1,2), Rational(1,2), coeff=sqrt(6)/3).
                add(ProductState.create(Rational(1,1), Rational(1,1), Rational(1,2), Rational(-1,2), coeff=sqrt(3)/3)),
                ProductState.create(Rational(1,1), Rational(-1,1), Rational(1,2), Rational(1,2), coeff=sqrt(3)/3).
                add(ProductState.create(Rational(1,1), Rational(0,1), Rational(1,2), Rational(-1,2), coeff=sqrt(6)/3)),
                ProductState.create(Rational(1,1), Rational(-1,1), Rational(1,2), Rational(-1,2)),
                ProductState.create(Rational(1,1), Rational(0,1), Rational(1,2), Rational(1,2), coeff=-sqrt(3)/3).
                add(ProductState.create(Rational(1,1), Rational(1,1), Rational(1,2), Rational(-1,2), coeff=sqrt(6)/3)),
                ProductState.create(Rational(1,1), Rational(-1,1), Rational(1,2), Rational(1,2), coeff=-sqrt(6)/3).
                add(ProductState.create(Rational(1,1), Rational(0,1), Rational(1,2), Rational(-1,2), coeff=sqrt(3)/3))]
    spinops1 = [ProductOp.S1x, ProductOp.S1y, ProductOp.S1z]
    spinops2 = [ProductOp.S2x, ProductOp.S2y, ProductOp.S2z]
    dmat = Matrix([[D11, D12, D13],
                   [D12, D22, D23],
                   [D13, D23, D33]])
    resmat = eye(len(eigstates), len(eigstates))
    for k, s1 in enumerate(eigstates):
        for l, s2 in enumerate(eigstates):
            tempres = Rational(0)
            for i in range(3):
                for j in range(3):
                    tempres += dmat[i,j]*s1.inner_product(spinops1[i](spinops1[j](s2)))
            term1 = s1.inner_product(spinops1[0](spinops2[0](s2)))
            term2 = s1.inner_product(spinops1[1](spinops2[1](s2)))
            term3 = s1.inner_product(spinops1[2](spinops2[2](s2)))
            tempres += J*(term1+term2+term3)
            resmat[k,l] = tempres
    return resmat


def modelHzfs2():
    eigstates = [SumState.create(Rational(3,2), Rational(3,2)),
                SumState.create(Rational(3,2), Rational(1,2)),
                SumState.create(Rational(3,2), Rational(-1,2)),
                SumState.create(Rational(3,2), Rational(-3,2)),
                SumState.create(Rational(1,2), Rational(1,2)),
                SumState.create(Rational(1,2), Rational(-1,2))]
    spinops1 = [SumOp.Sx, SumOp.Sy, SumOp.Sz]
    dmat = Matrix([[D11, D12, D13],
                   [D12, D22, D23],
                   [D13, D23, D33]])
    resmat = eye(len(eigstates), len(eigstates))
    for k, s1 in enumerate(eigstates):
        for l, s2 in enumerate(eigstates):
            tempres = Rational(0)
            for i in range(3):
                for j in range(3):
                    tempres += dmat[i,j]*s1.inner_product(spinops1[i](spinops1[j](s2)))
            resmat[k,l] = tempres
    return resmat

def modelHzfsmin():
    eigstates = [SumState.create(Rational(3,2), Rational(3,2)),
                SumState.create(Rational(3,2), Rational(1,2)),
                SumState.create(Rational(3,2), Rational(-1,2)),
                SumState.create(Rational(3,2), Rational(-3,2))]
    spinops1 = [SumOp.Sx, SumOp.Sy, SumOp.Sz]
    dmat = Matrix([[D11, D12, D13],
                   [D12, D22, D23],
                   [D13, D23, D33]])
    resmat = eye(len(eigstates), len(eigstates))
    for k, s1 in enumerate(eigstates):
        for l, s2 in enumerate(eigstates):
            tempres = Rational(0)
            for i in range(3):
                for j in range(3):
                    tempres += dmat[i,j]*s1.inner_product(spinops1[i](spinops1[j](s2)))
            resmat[k,l] = tempres
    return resmat

def getParams(symbmat: Matrix, projmat):
    eqlist = [Eq(t1, round(t2, 4)) for t1, t2 in zip(symbmat, projmat) if t1 != 0]
    paramlist = [D11, D12, D13, D22, D23, D33,
                                J11, J12, J13, J21, J22, J23, J31, J32, J33]
    mat, vec = linear_eq_to_matrix(eqlist, paramlist)
    #row reduced echelon form to eliminate dependent rows
    rref,inds = mat.T.rref()
    eqlist = [eq for i, eq in enumerate(eqlist) if i in inds]
    res = solve(eqlist, paramlist)
    return res

def getParamsmin(symbmat: Matrix, projmat):
    eqlist = [Eq(t1, round(t2, 4)) for t1, t2 in zip(symbmat, projmat) if t1 != 0]
    paramlist = [D11, D12, D13, D22, D23, D33]
    mat, vec = linear_eq_to_matrix(eqlist, paramlist)
    #row reduced echelon form to eliminate dependent rows
    rref,inds = mat.T.rref()
    eqlist = [eq for i, eq in enumerate(eqlist) if i in inds]
    for eq in eqlist:
        pprint(eq)
    res = solve(eqlist, paramlist)
    return res
    
def mininal_set(symbmat: Matrix, projmat):
    eqlist = [Eq(t1, round(t2, 4)) for t1, t2 in zip(symbmat, projmat) if t1 != 0]
    paramlist = [D11, D12, D13, D22, D23, D33,
                                J11, J12, J13, J21, J22, J23, J31, J32, J33]
    neqlist = []
    for eq in eqlist:
        lhsreal, lhsim = re(eq.lhs), im(eq.lhs)
        rhsreal, rhsim = re(eq.rhs), im(eq.rhs)
        eq1 = Eq(lhsreal, rhsreal)
        eq2 = Eq(lhsim, rhsim)
        if eq1 != True:
            neqlist.append(eq1)
        if eq2 != True:
            neqlist.append(eq2)
    mat, vec = linear_eq_to_matrix(neqlist, paramlist)
    rref,inds = mat.T.rref()
    neqlist = [eq for i, eq in enumerate(neqlist) if i in inds]
    return solve(neqlist, paramlist)



def mininal_set_min(symbmat: Matrix, projmat):
    eqlist = [Eq(t1, round(t2, 4)) for t1, t2 in zip(symbmat, projmat) if t1 != 0]
    paramlist = [D11, D12, D13, D22, D23, D33]
    neqlist = []
    for eq in eqlist:
        lhsreal, lhsim = re(eq.lhs), im(eq.lhs)
        rhsreal, rhsim = re(eq.rhs), im(eq.rhs)
        eq1 = Eq(lhsreal, rhsreal)
        eq2 = Eq(lhsim, rhsim)
        if eq1 != True:
            neqlist.append(eq1)
        if eq2 != True:
            neqlist.append(eq2)
    mat, vec = linear_eq_to_matrix(neqlist, paramlist)
    rref,inds = mat.T.rref()
    neqlist = [eq for i, eq in enumerate(neqlist) if i in inds]
    return solve(neqlist, paramlist)


def eliminate_duplicates(eqs):
    res = []
    for i, eq in enumerate(eqs):
        isduplicate = False
        for eq1 in eqs[i+1:]:
            if simplify(eq.lhs)==simplify(eq1.lhs) or simplify(eq.lhs)==simplify(-1*eq1.lhs):
                isduplicate = True
                break
        if not isduplicate:
            res.append(eq)
    return res

def projectedStates(allstates):
    res = [["|3/2, 3/2>", "|3/2, 1/2>", "|3/2, -1/2>", "|3/2, -3/2>", "|1/2, 1/2>", "|1/2, -1/2>"]]
    for state in allstates:
        temp = []
        for s in state.states:
            print(simplify(s.coeff), s.S, s.M)
        print("Energy = ", state.energy)
        res.append(temp)


def getDvalue(params):
    dmat = Matrix([[D11, D12, D13],
                   [D12, D22, D23],
                   [D13, D23, D33]])
    ndmat = dmat.subs(params)
    eigenvals = ndmat.eigenvals()
    eigenvecs = ndmat.eigenvects()
    veclist, eigenvals = ndmat.diagonalize()
    eiglist = np.array([eigenvals[i,i] for i in range(eigenvals.rows)])
    dval = eiglist[2] - 0.5*(eiglist[0]+eiglist[1])
    return dval, abs(0.5*(eiglist[0]-eiglist[1])/dval), ndmat.applyfunc(lambda x: round(x, 4)),\
          veclist
            
def getJmat(params):
    jmat = Matrix([[J11, J12, J13],
                    [J21, J22, J23],
                    [J31, J32, J33]])
    return jmat.subs(params).applyfunc(lambda x: round(x, 4))

def getSymAsym(exchtensor):
    anisosym = eye(3,3)
    antisym = eye(3,3)
    for i in range(3):
        for j in range(3):
            anisosym[i,j] = 0.5*(exchtensor[i,j]+exchtensor[j,i])
            antisym[i,j] = 0.5*(exchtensor[i,j]-exchtensor[j,i])
    return anisosym.applyfunc(lambda x: round(x, 4)), \
    Matrix([antisym[1,2], -antisym[0,2], antisym[0,1]]).applyfunc(lambda x: round(x, 4))

def get_overlap(eigenstates):
    res = []
    for e1 in eigenstates:
        temp = []
        for e2 in eigenstates:
            temp.append(simplify(e1.inner_product(e2)))
        res.append(temp)
    return np.array(res, dtype=np.complex128)

def states_to_mat(eigenstates:list[SumState], basis:list[SumBasis]):
    res = []
    ens = []
    for eig in eigenstates:
        res.append(eig.to_vec(basis))
        ens.append(eig.energy)
    return np.array(ens), np.array(res)


def mat_to_states(mat, basis:list[SumBasis], energies=None):
    res = []
    rc, cc = mat.shape
    for i in range(rc):
        temp = SumState.create_empty()
        for j in range(cc):
            temp.addstate(SumBasis(basis[j].S, basis[j].M, coeff=mat[i,j]))
        res.append(temp)
        if energies.any() != None:
            res[-1].energy = energies[i]
    return res
#The states are treated as row vectors instead of column vectors. Hence the weird reordering of operations
def deCloizeaux_orthogonalization(eigenstates, model_basis):
    ens, states = states_to_mat(eigenstates, model_basis)
    overlap = np.conj(states)@(states.T)
    s_inv = sp.linalg.fractional_matrix_power(overlap, -0.5)
    nstates = np.conj(s_inv)@states
    return mat_to_states(nstates, model_basis, energies=ens)

""" def states_to_mat(eigenstates:list[SumState], basis:list[SumBasis]):
    res = []
    ens = []
    for eig in eigenstates:
        res.append(eig.to_vec(basis))
        ens.append(eig.energy)
    return np.array(ens), np.array(res).T


def mat_to_states(mat, basis:list[SumBasis], energies=None):
    res = []
    rc, cc = mat.shape
    for i in range(cc):
        temp = SumState.create_empty()
        for j in range(rc):
            temp.addstate(SumBasis(basis[j].S, basis[j].M, coeff=mat[i,j]))
        res.append(temp)
        if energies.any() != None:
            res[-1].energy = energies[i]
    return res

def deCloizeaux_orthogonalization(eigenstates, model_basis):
    ens, states = states_to_mat(eigenstates, model_basis)
    overlap = states.conj().T@(states)
    s_inv = sp.linalg.fractional_matrix_power(overlap, -0.5)
    nstates = s_inv@states.conj().T
    print(s_inv)
    return mat_to_states(nstates, model_basis, energies=ens) """

def find_axis(dtensor):
    transform = eye(3,3)
    diagelms = [dtensor[i,i] for i in range(dtensor.rows)]
    sorted_indices = np.argsort(np.array(diagelms, dtype=np.float64))
    x,y = 0,1
    z = sorted_indices[-1]
    if diagelms[0]>diagelms[1]:
        pass
    elif diagelms[0]<diagelms[1]:
        x,y = y,x
    tempx = transform.col(x)
    tempy = transform.col(y)
    transform[:,0] = tempx
    transform[:,1] = tempy
    return transform, x,y,z


