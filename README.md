# AnisoExchange
Currently this package only supports the calculation of anisotrpic exchange tensors between two spin centers. Currently the spin centers have hard coded values of $S_1=1$ and $S_2=1/2$.
# Prerequisits
* numpy>=2.4.4
* sympy>=1.14.0
# Usage
`python3 exchparams.py <orca-output-file>` \
The `TPrint` value must be set to a very low value (<1e-9) in the ORCA input file
