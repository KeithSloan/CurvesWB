# -*- coding: utf-8 -*-

__title__ = "Nurbs tools"
__author__ = "Christophe Grellier (Chris_G)"
__license__ = "LGPL 2.1"
__doc__ = "Collection of tools for Nurbs."

import FreeCAD
import Part

#import _utils
#debug = _utils.debug
#debug = _utils.doNothing

def error(s):
    FreeCAD.Console.PrintError("%s\n"%s)

class BsplineBasis(object):
    """Computes basis functions of a bspline curve, and its derivatives"""
    def __init__(self):
        self.knots = [0.0, 0.0, 1.0, 1.0]
        self.degree = 1

    def find_span(self,u):
        """ Determine the knot span index.
        - input: parameter u (float)
        - output: the knot span index (int)
        Nurbs Book Algo A2.1 p.68
        """
        n = len(self.knots)-self.degree-1
        if u == self.knots[n+1]:
            return(n-1)
        low = self.degree
        high = n+1
        mid = int((low+high)/2)
        while (u < self.knots[mid] or u >= self.knots[mid+1]):
            if (u < self.knots[mid]):
                high = mid
            else:
                low = mid
            mid = int((low+high)/2)
        return(mid)

    def basis_funs(self, i, u):
        """ Compute the nonvanishing basis functions.
        - input: start index i (int), parameter u (float)
        - output: basis functions values N (list of floats)
        Nurbs Book Algo A2.2 p.70
        """
        N = [0. for x in range(self.degree+1)]
        N[0] = 1.0
        left = [0.0]
        right = [0.0]
        for j in range(1,self.degree+1):
            left.append(u-self.knots[i+1-j])
            right.append(self.knots[i+j]-u)
            saved = 0.0
            for r in range(j):
                temp = N[r] / (right[r+1] + left[j-r])
                N[r] = saved + right[r+1] * temp
                saved = left[j-r]*temp
            N[j] = saved
        return(N)

    def ders_basis_funs(self, i, u, n):
        """ Compute nonzero basis functions and their derivatives.
        First section is A2.2 modified to store functions and knot differences.
        - input: start index i (int), parameter u (float), number of derivatives n (int)
        - output: basis functions and derivatives ders (array2d of floats)
        Nurbs Book Algo A2.3 p.72
        """
        ders = [[0.0 for x in range(self.degree+1)] for y in range(n+1)]
        ndu = [[1.0 for x in range(self.degree+1)] for y in range(self.degree+1)] 
        ndu[0][0] = 1.0
        left = [0.0]
        right = [0.0]
        for j in range(1,self.degree+1):
            left.append(u-self.knots[i+1-j])
            right.append(self.knots[i+j]-u)
            saved = 0.0
            for r in range(j):
                ndu[j][r] = right[r+1] + left[j-r]
                temp = ndu[r][j-1] / ndu[j][r]
                ndu[r][j] = saved + right[r+1] * temp
                saved = left[j-r]*temp
            ndu[j][j] = saved

        for j in range(0,self.degree+1):
            ders[0][j] = ndu[j][self.degree]
        for r in range(0,self.degree+1):
            s1 = 0
            s2 = 1
            a = [[0.0 for x in range(self.degree+1)] for y in range(2)]
            a[0][0] = 1.0
            for k in range(1,n+1):
                d = 0.0
                rk = r-k
                pk = self.degree-k
                if r >= k:
                    a[s2][0] = a[s1][0] / ndu[pk+1][rk]
                    d = a[s2][0] * ndu[rk][pk]
                if rk >= -1:
                    j1 = 1
                else:
                    j1 = -rk
                if (r-1) <= pk:
                    j2 = k-1
                else:
                    j2 = self.degree-r
                for j in range(j1,j2+1):
                    a[s2][j] = (a[s1][j]-a[s1][j-1]) / ndu[pk+1][rk+j]
                    d += a[s2][j] * ndu[rk+j][pk]
                if r <= pk:
                    a[s2][k] = -a[s1][k-1] / ndu[pk+1][r]
                    d += a[s2][k] * ndu[r][pk]
                ders[k][r] = d
                j = s1
                s1 = s2
                s2 = j
        r = self.degree
        for k in range(1,n+1):
            for j in range(0,self.degree+1):
                ders[k][j] *= r
            r *= (self.degree-k)
        return(ders)

    def evaluate(self, u, d):
        """ Compute the derivative d of the basis functions.
        - input: parameter u (float), derivative d (int)
        - output: derivative d of the basis functions (list of floats)
        """
        n = len(self.knots)-self.degree-1
        f = [0.0 for x in range(n)]
        span = self.find_span(u)
        ders = self.ders_basis_funs(span, u, d)
        for i,val in enumerate(ders[d]):
            f[span-self.degree+i] = val
        return(f)

# This KnotVector class is equivalent to the following knotSeq* functions
# I am not sure what is best: a class or a set of independant functions ?

class KnotVector(object):
    """Knot vector object to use in Bsplines"""
    def __init__(self, v=[0.0, 1.0]):
        self._vector = v
        self._min_max()

    def __repr__(self):
        return("KnotVector(%s)"%str(self._vector))

    @property
    def vector(self):
        return(self._vector)

    @vector.setter
    def vector(self, v):
        self._vector = v
        self._vector.sort()
        self._min_max()

    def _min_max(self):
        """Compute the min and max values of the knot vector"""
        self.maxi = max(self._vector)
        self.mini = min(self._vector)

    def reverse(self):
        """Reverse the knot vector"""
        newknots = [(self.maxi + self.mini - k) for k in self._vector]
        newknots.reverse()
        self._vector = newknots

    def normalize(self):
        """Normalize the knot vector"""
        self.scale()

    def scale(self, length=1.0):
        """Scales the knot vector to a given length"""
        if length <= 0.0:
            error("scale error : bad value")
        else:
            ran = self.maxi - self.mini
            newknots = [length * (k-self.mini)/ran for k in self._vector]
            self._vector = newknots
            self._min_max()

    def reversed_param(self, pa):
        """Returns the image of the parameter when the knot vector is reversed"""
        newvec = KnotVector()
        newvec.vector = [self._vector[0], pa, self._vector[-1]]
        newvec.reverse()
        return(newvec.vector[1])

    def create_uniform(self, degree, nb_poles):
        """Create a uniform knotVector from given degree and Nb of poles"""
        if degree >= nb_poles:
            error("create_uniform : degree >= nb_poles")
        else:
            nb_int_knots = nb_poles - degree - 1
            start = [0.0 for k in range(degree+1)]
            mid = [float(k) for k in range(1,nb_int_knots+1)]
            end = [float(nb_int_knots+1) for k in range(degree+1)]
            self._vector = start + mid + end
            self._min_max()

    def get_mults(self):
        """Get the list of multiplicities of the knot vector"""
        no_duplicates = list(set(self._vector))
        return([self._vector.count(k) for k in no_duplicates])

    def get_knots(self):
        """Get the list of unique knots, without duplicates"""
        return(list(set(self._vector)))

# ---------------------------------------------------

def knotSeqReverse(knots):
    """Reverse a knot vector
    revKnots = knotSeqReverse(knots)"""
    ma = max(knots)
    mi = min(knots)
    newknots = [ma+mi-k for k in knots]
    newknots.reverse()
    return(newknots)

def knotSeqNormalize(knots):
    """Normalize a knot vector
    normKnots = knotSeqNormalize(knots)"""
    ma = max(knots)
    mi = min(knots)
    ran = ma-mi
    newknots = [(k-mi)/ran for k in knots]
    return(newknots)

def knotSeqScale(knots, length = 1.0, start = 0.0):
    """Scales a knot vector to a given length
    newknots = knotSeqScale(knots, length = 1.0)"""
    if length <= 0.0:
        error("knotSeqScale : length <= 0.0")
    else:
        ma = max(knots)
        mi = min(knots)
        ran = ma-mi
        newknots = [start+(length*(k-mi)/ran) for k in knots]
        return(newknots)

def paramReverse(pa,fp,lp):
    """Returns the image of parameter param when knot sequence [fp,lp] is reversed.
    newparam = paramReverse(param,fp,lp)"""
    seq = [fp,pa,lp]
    return(knotSeqReverse(seq)[1])

def createKnots(degree, nbPoles):
    """Create a uniform knotVector from given degree and Nb of poles
    knotVector = createKnots(degree, nbPoles)"""
    if degree >= nbPoles:
        error("createKnots : degree >= nbPoles")
    else:
        nbIntKnots = nbPoles - degree - 1
        start = [0.0 for k in range(degree+1)]
        mid = [float(k) for k in range(1,nbIntKnots+1)]
        end = [float(nbIntKnots+1) for k in range(degree+1)]
        return(start+mid+end)

def createKnotsMults(degree, nbPoles):
    """Create a uniform knotVector and a multiplicities list from given degree and Nb of poles
    knots, mults = createKnotsMults(degree, nbPoles)"""
    if degree >= nbPoles:
        error("createKnotsMults : degree >= nbPoles")
    else:
        nbIntKnots = nbPoles - degree - 1
        knots = [0.0] + [float(k) for k in range(1,nbIntKnots+1)] + [float(nbIntKnots+1)]
        mults = [degree+1] + [1 for k in range(nbIntKnots)] + [degree+1]
        return(knots, mults)

def knotSeqToKnotsMults(seq):
    """create knots and mults lists from a knot sequence."""
    knots = list()
    mults = list()
    for k in seq:
        if not k in knots:
            knots.append(k)
            mults.append(seq.count(k))
    return(knots,mults)

# ---------------------------------------------------

def bspline_copy(bs, reverse = False, scale = 1.0):
    """Copy a BSplineCurve, with knotvector optionally reversed and scaled
    newbspline = bspline_copy(bspline, reverse = False, scale = 1.0)"""
    # Part.BSplineCurve.buildFromPolesMultsKnots( poles, mults , knots, periodic, degree, weights, CheckRational )
    mults = bs.getMultiplicities()
    weights = bs.getWeights()
    poles = bs.getPoles()
    knots = bs.getKnots()
    perio = bs.isPeriodic()
    ratio = bs.isRational()
    if scale:
        knots = knotSeqScale(knots, scale)
    if reverse:
        mults.reverse()
        weights.reverse()
        poles.reverse()
        knots = knotSeqReverse(knots)
    bspline = Part.BSplineCurve()
    bspline.buildFromPolesMultsKnots(poles, mults , knots, perio, bs.Degree, weights, ratio)
    return(bspline)

def curvematch(c1, c2, par1, level=0, scale=1.0):
    '''Modifies the start of curve C2 so that it joins curve C1 at parameter par1
    - level (integer) is the level of continuity at join point (C0, G1, G2, G3, etc)
    - scale (float) is a scaling factor of the modified poles of curve C2
    newC2 = curvematch(C1, C2, par1, level=0, scale=1.0)'''
    #c1 = c1.toNurbs()
    #c2 = c2.toNurbs()
    len1 = c1.length()
    len2 = c2.length()
    # scale the knot vector of C2
    seq2 = knotSeqScale(c2.KnotSequence, 0.5 * abs(scale) * len2)
    # get a scaled / reversed copy of C1
    if scale < 0:
        bs1 = bspline_copy(c1, True, len1) # reversed
    else:
        bs1 = bspline_copy(c1, False, len1) # not reversed
    pt1 = c1.value(par1) # point on input curve C1
    par1 = bs1.parameter(pt1) # corresponding parameter on reversed / scaled curve bs1

    p1 = bs1.getPoles()
    basis1 = BsplineBasis()
    basis1.knots = bs1.KnotSequence
    basis1.degree = bs1.Degree
    
    p2 = c2.getPoles()
    basis2 = BsplineBasis()
    basis2.knots = seq2
    basis2.degree = c2.Degree
    
    # Compute the (level+1) first poles of C2
    l = 0
    while l <= level:
        #FreeCAD.Console.PrintMessage("\nDerivative %d\n"%l)
        ev1 = basis1.evaluate(par1,d=l)
        ev2 = basis2.evaluate(c2.FirstParameter,d=l)
        #FreeCAD.Console.PrintMessage("Basis %d - %r\n"%(l,ev1))
        #FreeCAD.Console.PrintMessage("Basis %d - %r\n"%(l,ev2))
        poles1 = FreeCAD.Vector()
        for i in range(len(ev1)):
            poles1 += 1.0*ev1[i]*p1[i]
        val = ev2[l]
        if val == 0:
            error("Zero !")
            break
        else:
            poles2 = FreeCAD.Vector()
            for i in range(l):
                poles2 += 1.0*ev2[i]*p2[i]
            np = (1.0*poles1-poles2)/val
            #FreeCAD.Console.PrintMessage("Moving P%d from (%0.2f,%0.2f,%0.2f) to (%0.2f,%0.2f,%0.2f)\n"%(l,p2[l].x,p2[l].y,p2[l].z,np.x,np.y,np.z))
            p2[l] = np
        l += 1
    nc = c2.copy()
    for i in range(len(p2)):
        nc.setPole(i+1,p2[i])
    return(nc)

class blendCurve(object):
    def __init__(self, e1 = None, e2 = None):
        if e1 and e2:
            self.edge1 = e1
            self.edge2 = e2
            self.param1 = e1.FirstParameter
            self.param2 = e2.FirstParameter
            self.cont1 = 0
            self.cont2 = 0
            self.scale1 = 1.0
            self.scale2 = 1.0
            self.Curve = Part.BSplineCurve()
            self.getChordLength()
            self.autoScale = True
            self.maxDegree = int(self.Curve.MaxDegree)
        else:
            error("blendCurve initialisation error")
    
    def getChord(self):
        v1 = self.edge1.valueAt(self.param1)
        v2 = self.edge2.valueAt(self.param2)
        ls = Part.LineSegment(v1,v2)
        return(ls)
    
    def getChordLength(self):
        ls = self.getChord()
        self.chordLength = ls.length()
        if self.chordLength < 1e-6:
            error("error : chordLength < 1e-6")
            self.chordLength = 1.0

    def compute(self):
        nbPoles = self.cont1 + self.cont2 + 2
        e = self.getChord()
        poles = e.discretize(nbPoles)
        degree = nbPoles - 1
        if degree > self.maxDegree:
            degree = self.maxDegree
        knots, mults = createKnotsMults(degree, nbPoles)
        weights = [1.0 for k in range(nbPoles)]
        be = Part.BSplineCurve()
        be.buildFromPolesMultsKnots(poles, mults , knots, False, degree, weights, False)
        c1 = self.edge1.Curve.toBSpline(self.edge1.FirstParameter, self.edge1.LastParameter)
        nc = curvematch(c1, be, self.param1, self.cont1, self.scale1)
        rev = bspline_copy(nc, True, False)
        c2 = self.edge2.Curve.toBSpline(self.edge2.FirstParameter, self.edge2.LastParameter)
        self.Curve = curvematch(c2, rev, self.param2, self.cont2, self.scale2)

    def getPoles(self):
        self.compute()
        return(self.Curve.getPoles())

    def shape(self):
        self.compute()
        return(self.Curve.toShape())

    def curve(self):
        self.compute()
        return(self.Curve)

# ---------------------------------------------------

def move_param(c,p1,p2):
    c1 = c.copy()
    c2 = c.copy()
    c1.segment(c.FirstParameter,float(p2))
    c2.segment(float(p2),c.LastParameter)
    #print("\nSegment 1 -> %r"%c1.getKnots())
    #print("Segment 2 -> %r"%c2.getKnots())
    knots1 = knotSeqScale(c1.getKnots(), p1-c.FirstParameter)
    knots2 = knotSeqScale(c2.getKnots(), c.LastParameter-p1)
    c1.setKnots(knots1)
    c2.setKnots(knots2)
    #print("New 1 -> %r"%c1.getKnots())
    #print("New 2 -> %r"%c2.getKnots())
    return(c1,c2)

def move_params(c,p1,p2):
    curves = list()
    p1.insert(0,c.FirstParameter)
    p1.append(c.LastParameter)
    p2.insert(0,c.FirstParameter)
    p2.append(c.LastParameter)
    for i in range(len(p1)-1):
        c1 = c.copy()
        c1.segment(p2[i],p2[i+1])
        knots1 = knotSeqScale(c1.getKnots(), p1[i+1]-p1[i], p1[i])
        print("%s -> %s"%(c1.getKnots(),knots1))
        c1.setKnots(knots1)
        curves.append(c1)
    return(curves)

def join_curve(c1,c2):
    c = Part.BSplineCurve()
    # poles (sequence of Base.Vector), [mults , knots, periodic, degree, weights (sequence of float), CheckRational]
    new_poles = c1.getPoles()
    new_poles.extend(c2.getPoles()[1:])
    new_weights = c1.getWeights()
    new_weights.extend(c2.getWeights()[1:])
    new_mults = c1.getMultiplicities()[:-1]
    new_mults.append(c1.Degree)
    new_mults.extend(c2.getMultiplicities()[1:])
    knots1 = c1.getKnots()
    sk = c2.getKnots()
    knots2 = [knots1[-1] - sk[0] + k for k in sk]
    new_knots = knots1
    new_knots.extend(knots2[1:])
    print("poles   -> %r"%new_poles)
    print("weights -> %r"%new_weights)
    print("mults   -> %r"%new_mults)
    print("knots   -> %r"%new_knots)
    c.buildFromPolesMultsKnots(new_poles, new_mults, new_knots, False, c1.Degree, new_weights, True)
    return(c)

def join_curves(curves):
    c0 = curves[0]
    for c in curves[1:]:
        c0 = join_curve(c0,c)
    return(c0)

def reparametrize(c, p1, p2):
    '''Reparametrize a BSplineCurve so that parameter p1 is moved to p2'''
    if not isinstance(p1,(list, tuple)):
        c1,c2 = move_param(c, p1, p2)
        c = join_curve(c1,c2)
        return(c)
    else:
        curves = move_params(c, p1, p2)
        c = join_curves(curves)
        return(c)

class Vector4d(object):
    def __init__(self, pt=(0,0,0,1)):
        if isinstance(pt, Vector4d):
            self.x  = float(pt.x)
            self.y  = float(pt.y)
            self.z  = float(pt.z)
            self.w  = float(pt.w)
        elif isinstance(pt, (list,tuple)):
            if len(pt) == 4:
                # pt = (x,y,z,w)
                self.x  = float(pt[0])
                self.y  = float(pt[1])
                self.z  = float(pt[2])
                self.w  = float(pt[3])
            elif len(pt) == 3:
                # pt = (x,y,z) -> w=1
                self.x  = float(pt[0])
                self.y  = float(pt[1])
                self.z  = float(pt[2])
                self.w  = 1.0
            elif len(pt) == 2:
                # pt = (Vector, w)
                self.x  = float(pt[0].x)
                self.y  = float(pt[0].y)
                self.z  = float(pt[0].z)
                self.w  = float(pt[1])
        else:
            self.x  = 0.
            self.y  = 0.
            self.z  = 0.
            self.w  = 1.
        self.update_homogeneous()

    def update_homogeneous(self):
        self.wx = float(self.x) * self.w
        self.wy = float(self.y) * self.w
        self.wz = float(self.z) * self.w
    def update_non_homogeneous(self):
        if self.w == 0:
            error("Weight is null.")
            return()
        self.x = float(self.wx) / self.w
        self.y = float(self.wy) / self.w
        self.z = float(self.wz) / self.w

    def __repr__(self):
        return("Vector4d((%s,%s,%s,%s))"%(self.x,self.y,self.z,self.w))

    def __str__(self):
        return("Vector4d((%s,%s,%s,%s)) -> (%s,%s,%s,%s)"%(self.x,self.y,self.z,self.w,self.wx,self.wy,self.wz,self.w))

    def __add__(self, pt4):
        newpt = Vector4d(self)
        newpt.wx += pt4.wx
        newpt.wy += pt4.wy
        newpt.wz += pt4.wz
        newpt.w  += pt4.w
        newpt.update_non_homogeneous()
        return(newpt)

    def __radd__(self, pt4):
        newpt = Vector4d(self)
        newpt.wx += pt4.wx
        newpt.wy += pt4.wy
        newpt.wz += pt4.wz
        newpt.w  += pt4.w
        newpt.update_non_homogeneous()
        return(newpt)

    def __sub__(self, pt4):
        newpt = Vector4d(self)
        newpt.wx -= pt4.wx
        newpt.wy -= pt4.wy
        newpt.wz -= pt4.wz
        newpt.w  -= pt4.w
        newpt.update_non_homogeneous()
        return(newpt)

    def __mul__(self, mu):
        newpt = Vector4d(self)
        m = float(mu)
        newpt.wx *= m
        newpt.wy *= m
        newpt.wz *= m
        newpt.w  *= m
        newpt.update_non_homogeneous()
        return(newpt)

    def __rmul__(self, mu):
        newpt = Vector4d(self)
        m = float(mu)
        newpt.wx *= m
        newpt.wy *= m
        newpt.wz *= m
        newpt.w  *= m
        newpt.update_non_homogeneous()
        return(newpt)

    def __div__(self, mu):
        if mu == 0:
            error("Division by zero.")
            return()
        newpt = Vector4d(self)
        m = float(mu)
        newpt.wx /= m
        newpt.wy /= m
        newpt.wz /= m
        newpt.w  /= m
        newpt.update_non_homogeneous()
        return(newpt)

    def distanceToPoint(self, other):
        x2 = pow(self.wx - other.wx, 2)
        y2 = pow(self.wy - other.wy, 2)
        z2 = pow(self.wz - other.wz, 2)
        w2 = pow(self.w  - other.w,  2)
        return(pow(x2 + y2 + z2 + w2, 0.5))
    
    def vector3d(self):
        return(FreeCAD.Vector(self.x, self.y, self.z))
    
    def weight(self):
        return(self.w)

def remove_knot(curve, index, dest_mult, tol):
    """ Remove knot of index 'index' to multiplicity 'dest_mult', with tolerance 'tol'.
    Nurbs Book Algo A5.8 p.185
    """
    # initialize the book algo variables
    #Pw = [Vector4d((z[0],z[1])) for z in zip(curve.getPoles(),curve.getWeights())]
    Pw = curve.getPoles()
    u = curve.getKnot(index)
    U = curve.KnotSequence
    s = int(curve.getMultiplicity(index))
    #r = U.index(index-1)
    r = len(U)-1-U[::-1].index(u) # index of the last knot u in KnotSequence U
    error("r=%d"%r)
    num = s - dest_mult
    n = int(curve.NbPoles)
    p = int(curve.Degree)
    temp = [0. for x in range(n)]
    
    # Start of the book algo
    m = n+p+1
    ord = p+1
    fout = (2*r-s-p)/2 # or int((2*r-s-p)/2)
    last = r-s
    first = r-p
    for t in range(num):
        error("t = %d"%t)
        off = first-1
        temp[0] = Pw[off]
        temp[last+1-off] = Pw[last+1]
        i = first
        j = last
        ii = 1
        jj = last - off
        remflag = 0
        while (j-i) > t:
            alfi = (u-U[i]) / (U[i+ord+t]-U[i])
            alfj = (u-U[j-t]) / (U[j+ord]-U[j-t])
            temp[ii] = (Pw[i] - (1.0-alfi) * temp[ii-1]) / alfi
            temp[jj] = (Pw[j] - alfj * temp[jj+1]) / (1.0-alfj)
            i = i+1
            ii = ii+1
            j = j-1
            jj = jj-1
        if (j-i) < t:
            if temp[ii-1].distanceToPoint(temp[jj+1]) <= tol:
                remflag = 1
        else:
            alfi = (u-U[i]) / (U[i+ord+t]-U[i])
            point = alfi*temp[ii+t+1] + (1.0-alfi)*temp[ii-1]
            if Pw[i].distanceToPoint(point) <= tol:
                remflag = 1
        if remflag == 0:
            break
        else:
            i = first
            j = last
            while (j-i) > t:
                Pw[i] = temp[i-off]
                Pw[j] = temp[j-off]
                i = i+1
                j = j-1
        first = first-1
        last = last+1
    if t == 0:
        return(None) #(t,U.__getslice__(0,len(U)-t),Pw.__getslice__(0,len(Pw)-t))
    for k in range(r+1,m): # -----> range(r+1,m+1):
        U[k-t] = U[k]
    j = fout
    i = j
    for k in range(1,t):
        if k%2 == 1:
            i = i+1
        else:
            j = j-1
    for k in range(i+1,n): # -----> range(i+1,n+1):
        Pw[j] = Pw[k]
        j = j+1
    knots, mults = knotSeqToKnotsMults(U.__getslice__(0,len(U)-t))
    return(t,knots, mults,Pw.__getslice__(0,len(Pw)-t))

def bs_remove_knot(curve, index, dest_mult, tol):
    t, k, m, p = remove_knot(curve, index, dest_mult, tol)
    bs = curve.copy()
    bs.buildFromPolesMultsKnots(p, m, k, curve.isPeriodic(), curve.Degree)
    return(bs)
            
                

def test():
    bb = BsplineBasis()
    bb.knots = [0.,0.,0.,0.,1.,2.,3.,3.,3.,3.]
    bb.degree = 3
    parm = 3.0

    span = bb.find_span(parm)
    print("Span index : %d"%span)
    f0 = bb.evaluate(parm,d=0)
    f1 = bb.evaluate(parm,d=1)
    f2 = bb.evaluate(parm,d=2)
    print("Basis functions    : %r"%f0)
    print("First derivatives  : %r"%f1)
    print("Second derivatives : %r"%f2)
    
    # compare to splipy results
    try:
        import splipy
    except ImportError:
        print("splipy is not installed.")
        return(False)
    
    basis1 = splipy.BSplineBasis(order=bb.degree+1, knots=bb.knots)
    
    print("splipy results :")
    print(basis1.evaluate(parm,d=0).A1.tolist())
    print(basis1.evaluate(parm,d=1).A1.tolist())
    print(basis1.evaluate(parm,d=2).A1.tolist())

