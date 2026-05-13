from __future__ import annotations
from sympy import symbols, Matrix, Rational, sqrt, I, simplify, conjugate
import copy
import logging
import numpy as np


class SumBasis:
    S=0
    M=0
    coeff = Rational(1)
    def __init__(self, S, M, coeff=Rational(1)):
        self.S = S
        self.M = M
        self.coeff = coeff
    
    def inner_product(self, state):
        if self.S == state.S and self.M == state.M:
            return conjugate(self.coeff)*state.coeff
        return Rational(0)
    
    def is_equal(self, basis):
        if self.S == basis.S and self.M == basis.M:
            return True
        return False
    
class SumState:
    states = []
    energy = 0.0
    def __init__(self):
        self.states = []

    @staticmethod
    def create(S, M, coeff=Rational(1), energy=0.0):
        res = SumState()
        bfunc = SumBasis(S, M, coeff=coeff)
        res.addstate(bfunc)
        res.energy = energy
        return res
    
    @staticmethod
    def create_empty():
        res = SumState()
        return res

    def add(self, state):
        res = copy.deepcopy(self)
        for s in state.states:
            res.states.append(s)
        return res

    def addstate(self, state):
        self.states.append(state)

    def sub(self, state):
        for s in state.states:
            s.coeff *= -1
            self.states.append(s)

    def scale(self, num):
        res = copy.deepcopy(self)
        for s in res.states:
            s.coeff *= num
        return res

    def inner_product(self, state):
        res = Rational(0)
        for s1 in self.states:
            for s2 in state.states:
                if s1.S == s2.S and s1.M == s2.M:
                    res += simplify(conjugate(s1.coeff)*s2.coeff)
        return res
                
    def stringify(self):
        res = ""
        for s in self.states[:-1]:
            if s.coeff==0: continue
            res += f"({str(s.coeff)})"
            res += f"|{s.S}, {s.M}> + "
        return res + (f"({self.states[-1].coeff})|{self.states[-1].S}, {self.states[-1].M}>" if self.states[-1].coeff !=0 else "")
    
    def to_vec(self, basis:list[SumBasis]):
        res = np.zeros(len(basis), dtype=np.complex128)
        for i, b in enumerate(basis):
            for s in self.states:
                if b.is_equal(s):
                    res[i] = s.coeff
        return res

    

class ProductBasis:
    s1, s2, m1, m2 = 0, 0, 0, 0
    coeff = 1.0
    def __init__(self, s1, m1, s2, m2, coeff=Rational(1)):
        if s1<s2:
            logging.debug(f"Error: s1 should be greater than s2! s1={s1}, s2={s2}")
            exit(0)
        self.s1, self.s2, self.m1, self.m2 = s1, s2, m1, m2
        self.coeff = coeff

    def inner_product(self, state)-> Rational | float | int:
        if self.s1 == state.s1 and self.s2 == state.s2 and self.m1 == state.m1 and self.m2 == state.m2:
            return simplify(conjugate(self.coeff)*state.coeff)
        return Rational(0)
    
    def is_equal(self, basis):
        if self.s1 == basis.s1 and self.s2 == basis.s2 and self.m1 == basis.m1 and self.m2 == basis.m2:
            return True
        return False
    
    def aspstate(self):
        return ProductState.create(self.s1, self.m1, self.s2, self.m2, coeff=self.coeff)

class ProductState:
    states: list[ProductBasis] = []
    def __init__(self, s1, s2, m1, m2):
        self.states = []

    @staticmethod
    def create(s1, m1, s2, m2, coeff=Rational(1))->ProductState:
        res = ProductState(0, 0, 0, 0)
        bfunc = ProductBasis(s1, m1, s2, m2, coeff=coeff)
        res.addstate(bfunc)
        return res

    def addstate(self, state)->None:
        self.states.append(state)

    def add(self, state)->ProductState:
        res = copy.deepcopy(self)
        for s in state.states:
            res.states.append(s)
        return res
            

    def sub(self, state)->ProductState:
        res = copy.deepcopy(self)
        for s in state.states:
            s.coeff *= -1
            res.states.append(s)
        return res

    def inner_product(self, state):
        res = Rational(0)
        for st1 in self.states:
            for st2 in state.states:
                if st1.s1 == st2.s1 and st1.s2 == st2.s2 and st1.m1 == st2.m1 and st1.m2 == st2.m2:
                    res += simplify(conjugate(st1.coeff)*st2.coeff)
        return res
    
    def scale(self, num)->ProductState:
        res = copy.deepcopy(self)
        for s in res.states:
            s.coeff *= num
        return res

    def stringify(self):
        res = ""
        for s in self.states[:-1]:
            res += str(s.coeff)
            res += f"({s.s1}, {s.m1}; {s.s2}, {s.m2}) + "
        return res + f"{self.states[-1].coeff}({self.states[-1].s1}, {self.states[-1].m1}; {self.states[-1].s2}, {self.states[-1].m2})"
    
    def to_vec(self, basis:list[ProductBasis]):
        res = np.zeros(len(basis))
        for i, b in enumerate(basis):
            for s in self.states:
                if b.is_equal(s):
                    res[i] = s.coeff
        return res
    
class SumOp:

    def Sz(state: SumState):
        res = SumState()
        for s in state.states:
            temp = s.coeff*s.M
            res.addstate(SumBasis(s.S, s.M, coeff=temp))
        return res
    
    def Sx(state: SumState):
        res = SumState()
        for s in state.states:
            craise = sqrt(s.S*(s.S+1)-s.M*(s.M+1))
            clower = sqrt(s.S*(s.S+1)-s.M*(s.M-1))
            temp1  = s.coeff*craise*Rational(1,2)
            temp2  = s.coeff*clower*Rational(1,2)
            if s.M+1>s.S:
                temp1 = Rational(0)
            if s.M-1<-s.S:
                temp2 = Rational(0)
            sraise = SumBasis(s.S, s.M+1, coeff=temp1)
            slower = SumBasis(s.S, s.M-1, coeff=temp2)
            res.addstate(sraise)
            res.addstate(slower)
        return res
    
    def Sy(state: SumState):
        res = SumState()
        for s in state.states:
            craise = sqrt(s.S*(s.S+1)-s.M*(s.M+1))
            clower = sqrt(s.S*(s.S+1)-s.M*(s.M-1))
            temp1  = -s.coeff*craise*Rational(1,2)*I
            temp2  = s.coeff*clower*Rational(1,2)*I
            if s.M+1>s.S:
                temp1 = Rational(0)
            if s.M-1<-s.S:
                temp2 = Rational(0)
            sraise = SumBasis(s.S, s.M+1, coeff=temp1)
            slower = SumBasis(s.S, s.M-1, coeff=temp2)
            res.addstate(sraise)
            res.addstate(slower)
        return res


class ProductOp:

    def S1z(state: ProductState)->ProductState:
        res = ProductState(0, 0, 0, 0)
        for s in state.states:
            temp = s.coeff*s.m1
            res.addstate(ProductBasis(s.s1, s.m1, s.s2, s.m2, coeff=temp))
        return res
    
    def S2z(state: ProductState)->ProductState:
        res = ProductState(0, 0, 0, 0)
        for s in state.states:
            temp = s.coeff*s.m2
            res.addstate(ProductBasis(s.s1, s.m1, s.s2, s.m2, coeff=temp))
        return res

    def S1x(state: ProductState)->ProductState:
        res = ProductState(0, 0, 0, 0)
        for s in state.states:
            craise = sqrt(s.s1*(s.s1+1)-s.m1*(s.m1+1))
            clower = sqrt(s.s1*(s.s1+1)-s.m1*(s.m1-1))
            temp1  = s.coeff*craise*Rational(1,2)
            temp2  = s.coeff*clower*Rational(1,2)
            if s.m1+1>s.s1:
                temp1 = Rational(0)
            if s.m1-1<-s.s1:
                temp2 = Rational(0)
            sraise = ProductBasis(s.s1, s.m1+1, s.s2, s.m2, coeff=temp1)
            slower = ProductBasis(s.s1, s.m1-1, s.s2, s.m2, coeff=temp2)
            res.addstate(sraise)
            res.addstate(slower)
        return res
    
    def S2x(state: ProductState)->ProductState:
        res = ProductState(0, 0, 0, 0)
        for s in state.states:
            craise = sqrt(s.s2*(s.s2+1)-s.m2*(s.m2+1))
            clower = sqrt(s.s2*(s.s2+1)-s.m2*(s.m2-1))
            temp1  = s.coeff*craise*Rational(1,2)
            temp2  = s.coeff*clower*Rational(1,2)
            if s.m2+1>s.s2:
                temp1 = Rational(0)
            if s.m2-1<-s.s2:
                temp2 = Rational(0)
            sraise = ProductBasis(s.s1, s.m1, s.s2, s.m2+1, coeff=temp1)
            slower = ProductBasis(s.s1, s.m1, s.s2, s.m2-1, coeff=temp2)
            res.addstate(sraise)
            res.addstate(slower)
        return res
    
    def S1y(state: ProductState)->ProductState:
        res = ProductState(0, 0, 0, 0)
        for s in state.states:
            craise = sqrt(s.s1*(s.s1+1)-s.m1*(s.m1+1))
            clower = sqrt(s.s1*(s.s1+1)-s.m1*(s.m1-1))
            temp1  = -s.coeff*craise*Rational(1,2)*I
            temp2  = s.coeff*clower*Rational(1,2)*I
            if s.m1+1>s.s1:
                temp1 = Rational(0)
            if s.m1-1<-s.s1:
                temp2 = Rational(0)
            sraise = ProductBasis(s.s1, s.m1+1, s.s2, s.m2, coeff=temp1)
            slower = ProductBasis(s.s1, s.m1-1, s.s2, s.m2, coeff=temp2)
            res.addstate(sraise)
            res.addstate(slower)
        return res
    
    def S2y(state: ProductState)->ProductState:
        res = ProductState(0, 0, 0, 0)
        for s in state.states:
            craise = sqrt(s.s2*(s.s2+1)-s.m2*(s.m2+1))
            clower = sqrt(s.s2*(s.s2+1)-s.m2*(s.m2-1))
            temp1  = -s.coeff*craise*Rational(1,2)*I
            temp2  = s.coeff*clower*Rational(1,2)*I
            if s.m2+1>s.s2:
                temp1 = Rational(0)
            if s.m2-1<-s.s2:
                temp2 = Rational(0)
            sraise = ProductBasis(s.s1, s.m1, s.s2, s.m2+1, coeff=temp1)
            slower = ProductBasis(s.s1, s.m1, s.s2, s.m2-1, coeff=temp2)
            res.addstate(sraise)
            res.addstate(slower)
        return res

    def OpCross(state: ProductState)->tuple[ProductState, ...]:
        xcomp = ProductOp.S1y(ProductOp.S2z(state)).sub(ProductOp.S2y(ProductOp.S1z(state)))
        ycomp = ProductOp.S2x(ProductOp.S1z(state)).sub(ProductOp.S1x(ProductOp.S2z(state)))
        zcomp = ProductOp.S1x(ProductOp.S2y(state)).sub(ProductOp.S2x(ProductOp.S1y(state)))
        return (xcomp, ycomp, zcomp)
