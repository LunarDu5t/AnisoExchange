from __future__ import annotations
from qpack import *
from hamiltonian import *
import sys


def get_test_el(instates):
    s1 = SumState.create(Rational(3,2), Rational(-1,2))
    s2 = SumState.create(Rational(1,2), Rational(-1,2))
    res = 0
    term = SumState()
    for ps in instates:
        ncoeff = ps.inner_product(s2)
        term = term.add(ps.scale(ncoeff*ps.energy))
    res = simplify(s1.inner_product(term))
    return res

def get_min_state(instates):
    for ps in instates:
        for s in ps.states:
            if s.S==Rational(1,2) and s.M==Rational(-1,2):
                print(s.coeff)
        print("")

def gen_vecs(center, veclen, vecs, colors = ["red", "green", "blue"], wcyl = 0.08, whd = 0.12):
    rc, cc = vecs.shape
    for c in range(cc):
        arrowbodybase = center
        arrowbodytop = veclen*vecs.col(c)+center
        arrowheadbase = arrowbodytop
        arrowheadtop = arrowbodytop + veclen/4*vecs.col(c)
        print(f"draw color {colors[c]};")
        print(f"draw cylinder  {{{arrowbodybase[0]}    {arrowbodybase[1]}   {arrowbodybase[2]} }} {{{arrowbodytop[0]}  {arrowbodytop[1]} {arrowbodytop[2]} }} resolution 50 radius {wcyl};")
        print(f"draw cone  {{ {arrowheadbase[0]}  {arrowheadbase[1]} {arrowheadbase[2]} }} {{ {arrowheadtop[0]}  {arrowheadtop[1]} {arrowheadtop[2]}}} resolution 50 radius {whd}")

eigvecs = readeigvecs(sys.argv[1])
print("Norm of unnormalized eigenstates")
for v in eigvecs:
    print(sqrt(v.inner_product(v)))
eigvecs = deCloizeaux_orthogonalization(eigvecs, [SumBasis(Rational(3,2), Rational(3,2)),
                SumBasis(Rational(3,2), Rational(1,2)),
                SumBasis(Rational(3,2), Rational(-1,2)),
                SumBasis(Rational(3,2), Rational(-3,2)),
                SumBasis(Rational(1,2), Rational(1,2)),
                SumBasis(Rational(1,2), Rational(-1,2))])

#print("The 18th element is:  ", get_test_el(eigvecs))
#get_min_state(eigvecs)
#exit()

hmat = modelHaniso()
for r in range(hmat.rows):
    for c in range(hmat.cols):
        if c>=r:
            print(f"H_{{{r+1},{c+1}}}&="+latex(hmat[r,c])+"\\\\")
#hmat2 = modelHzfsJ()
projmat = projectedH(eigvecs)
print(latex(projmat.applyfunc(lambda x: round(x, 4))))
#pprint(hmat[2,5])
params = mininal_set(hmat, projmat)
dval, ebyd, dtensor, magnetic_axes = getDvalue(params)

print("Value of all parameters")
print(params)
print("D value of Ni center: ", dval)
print("E/D: ", ebyd)
print("Exchange tensor in molecular frame\n"+"-"*30)
jtensor = getJmat(params)
pprint(jtensor)
print("ZFS tensor\n"+"-"*30)
pprint(dtensor)
print("Symmetric anisotropic exchange tensor\n"+"-"*30)
symaniso, dmi = getSymAsym(jtensor)
pprint(symaniso)
#print(symaniso)
print("DMI vector\n"+"-"*30)
pprint(dmi)
ndtensor = magnetic_axes.inv()@dtensor@magnetic_axes
axis_transform, x,y,z = find_axis(ndtensor)
magaxes = axis_transform@magnetic_axes
""" print("Symmetric anisotropic exchange in magnetic frame")
njtensor = magaxes.inv()@jtensor@magaxes
pprint(njtensor.applyfunc(lambda x: round(x,4)))"""
print("Ordering of magnetic axis\n"+"-"*30)
print(f"x={x}, y={y}, z={z}")
print("Magnetic Axes\n"+"-"*30)
pprint(magaxes)

