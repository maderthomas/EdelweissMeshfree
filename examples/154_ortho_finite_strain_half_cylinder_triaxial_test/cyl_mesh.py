#!/usr/bin/env python3
"""
Butterfly (O-grid) hex mesh for a SOLID CYLINDER, emitted as EdelweissFE
`*node` / `*element` / `*nSet` blocks.

Why not a generator already in EdelweissFE:  `boxGen` is a box, and `pipeGen`
makes a HOLLOW pipe -- setting its inner radius to zero collapses the innermost
element layer onto the axis (zero-Jacobian wedges).  A solid cylinder needs an
O-grid: a structured core block surrounded by radial rings, so that no element
degenerates on the axis.

--------------------------------------------------------------------------------
GEOMETRY / TOPOLOGY
--------------------------------------------------------------------------------
The cylinder axis is **x** (so the loading axis matches the tensile bar of
paper_sect4_tension.py and the same x-z mid-slice contour views can be used).
The cross-section lives in the (y, z) plane, centred on the axis:

        core square [-a, a]^2 with nC x nC cells,  a = CORE_FRAC * R
        + nR radial ring layers blending the square boundary onto the circle

    z
    ^      ___------___            outer boundary: exact circle of radius R
    |    /   |  |  |   \\           (ring nodes at t = 1 sit ON it)
    |   /----+--+--+----\\
    |   |    |  |  |    |          core: structured nC x nC block
    |   |----+--+--+----|
    |   \\----+--+--+----/
    |    \\___|__|__|___/
    +-------------------> y

Perimeter index p = 0 .. 4*nC-1 walks the core-square boundary counter-clockwise
starting at the corner (y, z) = (a, -a), i.e. at angle -45 deg.  Ring node p of
layer j is the linear blend

    P(j, p) = (1 - t) * S_p + t * R * (cos th_p, sin th_p),   t = j / nR
    th_p    = -45 deg + p * 360 deg / (4 nC)

so square corners map to the circle at 45 deg (radially), the outer ring is
UNIFORMLY spaced in angle, and t = 1 reproduces the circle exactly.

--------------------------------------------------------------------------------
HEX NODE ORDERING  (must match what Marmot expects; boxGen is the reference)
--------------------------------------------------------------------------------
Decoding boxGen's 8-node connectivity (generators/boxgen.py:187) with
(ix, iy, iz) the low corner and iz the fastest index gives

    1:(ix,iy,iz)  2:(ix,iy,iz+1)  3:(ix+1,iy,iz+1)  4:(ix+1,iy,iz)
    5:(ix,iy+1,iz) 6:(ix,iy+1,iz+1) 7:(ix+1,iy+1,iz+1) 8:(ix+1,iy+1,iz)

i.e. local xi = 1->2, eta = 1->4, zeta = 1->5, and det J = (xi x eta).zeta > 0.
Transposed to "cross-section quad extruded along the axis", the rule is:

    nodes 1-4 = the cross-section quad at x_k, ordered CCW in the (y, z) plane
                (right-handed about +x)
    nodes 5-8 = the SAME four corners at x_{k+1}

because then xi x eta = (+y) x (+z) = +x = zeta.  `check()` below verifies the
resulting Jacobians numerically rather than trusting this derivation.
"""
import math

import numpy as np


# Standard 8-node hex isoparametric corner coordinates (Abaqus / Marmot ordering):
# xi = 1->2, eta = 1->4, zeta = 1->5.  Decoding boxGen with this gives xi=+z, eta=+x,
# zeta=+y and det J = (z x x).y = +1 > 0, confirming it is the convention in use.
_LOCAL = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                   [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)


class CylinderMesh:
    """Solid-cylinder O-grid hex mesh, axis along x from x=0 to x=L."""

    def __init__(self, R=1.25, L=5.0, nC=2, nR=1, nX=10, coreFrac=0.5, half=False):
        if nC % 2:
            raise ValueError("nC must be EVEN so that a node plane sits at y=0 and z=0 "
                             "(needed for the axis nSets and the mid-plane contour slice)")
        self.R, self.L, self.nC, self.nR, self.nX = R, L, nC, nR, nX
        self.a = coreFrac * R
        self.half = half
        self._build()
        if half:
            self._keepHalf()

    def _keepHalf(self):
        """Discard the z < 0 half, leaving the z = 0 plane as a symmetry plane.

        Mader et al. (2022) Sec. 4.3 exploit exactly this symmetry: the stratification planes are
        inclined by beta about an axis lying in the symmetry plane, so the expected failure mode is
        symmetric and only half the specimen is discretised.  No element straddles z = 0: for EVEN
        nC the core grid has a node plane at z = 0, and the ring nodes include the angles 0 and 180
        deg (p = nC/2 and 3nC/2), so the cut is exact and needs no tolerance beyond roundoff.
        """
        cen = np.array([self.coords[el].mean(axis=0) for el in self.elements])
        keep = np.where(cen[:, 2] > 1e-9)[0]
        els = [self.elements[i] for i in keep]
        used = sorted({n for el in els for n in el})
        remap = {old: new for new, old in enumerate(used)}
        self.coords = self.coords[used]
        self.elements = [[remap[n] for n in el] for el in els]

    # ------------------------------------------------------------------ mesh
    def _perimeter_core_ij(self, p):
        """perimeter index -> (iy, iz) on the core grid, CCW from the corner (a, -a)"""
        nC = self.nC
        e, k = divmod(p, nC)
        return [(nC, k), (nC - k, nC), (0, nC - k), (k, 0)][e]

    def _build(self):
        R, a, nC, nR, nX = self.R, self.a, self.nC, self.nR, self.nX
        sec = []                       # (y, z) of every cross-section node
        self.coreId = {}               # (iy, iz) -> section-node index
        for iy in range(nC + 1):
            for iz in range(nC + 1):
                self.coreId[(iy, iz)] = len(sec)
                sec.append((-a + 2 * a * iy / nC, -a + 2 * a * iz / nC))

        self.ringId = {}               # (layer j>=1, p) -> section-node index
        nP = 4 * nC
        for j in range(1, nR + 1):
            t = j / nR
            for p in range(nP):
                sy, sz = sec[self.coreId[self._perimeter_core_ij(p)]]
                th = math.radians(-45.0 + p * 360.0 / nP)
                cy, cz = R * math.cos(th), R * math.sin(th)
                self.ringId[(j, p)] = len(sec)
                sec.append(((1 - t) * sy + t * cy, (1 - t) * sz + t * cz))
        self.sec = np.array(sec)
        self.nSec = len(sec)

        # cross-section quads, each CCW in (y, z)
        quads = []
        for iy in range(nC):
            for iz in range(nC):
                quads.append([self.coreId[(iy, iz)], self.coreId[(iy + 1, iz)],
                              self.coreId[(iy + 1, iz + 1)], self.coreId[(iy, iz + 1)]])

        def ring_node(j, p):
            p %= nP
            return self.coreId[self._perimeter_core_ij(p)] if j == 0 else self.ringId[(j, p)]

        for j in range(nR):
            for p in range(nP):
                # (radial, tangential) is right-handed about +x -> this order is CCW
                quads.append([ring_node(j, p), ring_node(j + 1, p),
                              ring_node(j + 1, p + 1), ring_node(j, p + 1)])
        self.quads = quads

        # extrude along x
        self.x = np.linspace(0.0, self.L, nX + 1)
        self.coords = np.zeros((self.nSec * (nX + 1), 3))
        for k in range(nX + 1):
            sl = slice(k * self.nSec, (k + 1) * self.nSec)
            self.coords[sl, 0] = self.x[k]
            self.coords[sl, 1:] = self.sec
        self.elements = [[k * self.nSec + q[i] for i in range(4)]
                         + [(k + 1) * self.nSec + q[i] for i in range(4)]
                         for k in range(nX) for q in quads]

    # ------------------------------------------------------------- validation
    def jacobians(self):
        """min det J over the 8 corners of every element (trilinear hex)."""
        g = 1.0 / math.sqrt(3.0)
        gp = [(sx * g, sy * g, sz * g) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        # local corner coordinates in the ordering documented above
        L = _LOCAL
        out = []
        for el in self.elements:
            X = self.coords[el]
            m = np.inf
            for xi, eta, ze in gp:
                dN = np.empty((8, 3))
                for i, (a_, b_, c_) in enumerate(L):
                    dN[i] = [0.125 * a_ * (1 + b_ * eta) * (1 + c_ * ze),
                             0.125 * b_ * (1 + a_ * xi) * (1 + c_ * ze),
                             0.125 * c_ * (1 + a_ * xi) * (1 + b_ * eta)]
                m = min(m, np.linalg.det(X.T @ dN))
            out.append(m)
        return np.array(out)

    def volume(self):
        """sum of 2x2x2 Gauss-integrated element volumes"""
        g = 1.0 / math.sqrt(3.0)
        L = _LOCAL
        tot = 0.0
        for el in self.elements:
            X = self.coords[el]
            for sx in (-g, g):
                for sy in (-g, g):
                    for sz in (-g, g):
                        dN = np.empty((8, 3))
                        for i, (a_, b_, c_) in enumerate(L):
                            dN[i] = [0.125 * a_ * (1 + b_ * sy) * (1 + c_ * sz),
                                     0.125 * b_ * (1 + a_ * sx) * (1 + c_ * sz),
                                     0.125 * c_ * (1 + a_ * sx) * (1 + b_ * sy)]
                        tot += np.linalg.det(X.T @ dN)
        return tot

    def area(self):
        """EXACT cross-sectional area of the MESHED domain (the cross-section is prismatic).

        The O-grid's outer boundary is the polygon inscribed in the circle, so the meshed
        area is (4 nC / 2) R^2 sin(2 pi / 4 nC) < pi R^2 -- 9.97% low for nC = 2 (octagon),
        2.55% for nC = 4.  Nominal stresses MUST be formed with this area, not pi R^2,
        otherwise every strength carries that discretisation bias.
        """
        return self.volume() / self.L

    # ------------------------------------------------------------------ nSets
    def nsets(self, tol=1e-9):
        c = self.coords
        left = np.where(np.abs(c[:, 0]) < tol)[0]
        right = np.where(np.abs(c[:, 0] - self.L) < tol)[0]
        onAxis = np.where((np.abs(c[:, 1]) < tol) & (np.abs(c[:, 2]) < tol))[0]
        # left-face node on the +y axis: kills rotation about x (u_z = 0 there)
        yPos = np.where((np.abs(c[:, 0]) < tol) & (c[:, 1] > tol) & (np.abs(c[:, 2]) < tol))[0]
        sym = np.where(np.abs(c[:, 2]) < tol)[0]            # the z = 0 symmetry plane
        # one node to pin the remaining y translation: on the axis at the bottom face
        pinY = np.array([i for i in onAxis if abs(c[i, 0]) < tol])[:1]
        return {"all": np.arange(len(c)), "left": left, "right": right,
                "ctr": np.array([i for i in onAxis if abs(c[i, 0]) < tol]),
                "rot": yPos[:1], "sym": sym, "pinY": pinY}

    # ------------------------------------------------------------------ output
    def centroids(self):
        return np.array([self.coords[el].mean(axis=0) for el in self.elements])

    def seed_elements(self, mode="sphere", radius=None, halfwidth=None, offset=0.0):
        """Elements of a weak IMPERFECTION at mid-height, used to nucleate a band.

        A homogeneous, statically determinate specimen has no reason to localise -- every
        material point is on the same stress path, so the response stays uniform to
        round-off (measured 5e-13 in the unseeded runs).  A band needs a perturbation, and
        per the README it must perturb the PLASTIC response: damage-parameter seeds are
        invisible here because the damage gate (alphaP >= 1) is crossed post hoc.  The
        caller therefore gives these elements a card with all four STRENGTHS scaled down,
        which keeps the yield surface self-similar.

        mode="sphere" : compact inclusion at the centre -> the band picks its own
                        inclination (what you want for a SHEAR band)
        mode="slab"   : full cross-section slice -> forces a band normal to the axis
        offset shifts the sphere off the axis in +y, which breaks the axisymmetry and
        stops the model from having to choose between a cone and a single band.
        """
        c = self.centroids()
        ctr = np.array([self.L / 2, offset, 0.0])
        if mode == "slab":
            hw = halfwidth if halfwidth is not None else self.L / self.nX
            return np.where(np.abs(c[:, 0] - self.L / 2) < hw)[0]
        r = radius if radius is not None else 0.25 * self.R
        return np.where(np.linalg.norm(c - ctr, axis=1) < r)[0]

    # -------------------------------------------------------- faces / surfaces
    #  EdelweissFE face IDs follow boxGen (generators/boxgen.py:454-461), whose local hex
    #  connectivity decodes (see the header) to
    #      face 1 = {1,2,3,4}   face 2 = {5,6,7,8}   face 3 = {1,2,6,5}
    #      face 4 = {2,3,7,6}   face 5 = {3,4,8,7}   face 6 = {1,4,8,5}
    #  In the cross-section-extruded ordering used here, local nodes 1-4 are the section quad
    #  at x_k and 5-8 the same quad at x_{k+1}, so:
    #      face 1 = the low-x end face,  face 2 = the high-x end face,
    #      face 4 = the face spanned by section-quad corners 2-3 extruded.
    #  For an OUTERMOST-RING element the quad is
    #      [ring(nR-1,p), ring(nR,p), ring(nR,p+1), ring(nR-1,p+1)]
    #  so corners 2 and 3 are the two nodes sitting exactly on the circle, and face 4 is the
    #  curved lateral face.  `check_faces()` verifies that numerically instead of trusting it.
    FACE_LOW_X, FACE_HIGH_X, FACE_LATERAL = 1, 2, 4
    _FACE_NODES = {1: (0, 1, 2, 3), 2: (4, 5, 6, 7), 3: (0, 1, 5, 4),
                   4: (1, 2, 6, 5), 5: (2, 3, 7, 6), 6: (0, 3, 7, 4)}

    def _faceNodes(self, el, fid):
        return [el[i] for i in self._FACE_NODES[fid]]

    def lateral_elements(self):
        """1-based labels of the elements whose face 4 lies on the cylinder surface.

        Determined GEOMETRICALLY (all four face-4 nodes at radius R) rather than from the element
        numbering, so it survives the half-model filter.  A confining pressure needs this; without
        it the cylinder mesh could only be loaded on its end faces.
        """
        out = []
        for k, el in enumerate(self.elements):
            X = self.coords[self._faceNodes(el, self.FACE_LATERAL)]
            if np.all(np.abs(np.hypot(X[:, 1], X[:, 2]) - self.R) < 1e-8):
                out.append(k + 1)
        return out

    def end_elements(self, high=False):
        """1-based labels of the elements with a face on the x = L (high) or x = 0 end."""
        xw = self.L if high else 0.0
        fid = self.FACE_HIGH_X if high else self.FACE_LOW_X
        out = []
        for k, el in enumerate(self.elements):
            X = self.coords[self._faceNodes(el, fid)]
            if np.all(np.abs(X[:, 0] - xw) < 1e-8):
                out.append(k + 1)
        return out

    def check_faces(self):
        """Numerically verify the three face IDs above.

        Returns (minRadialDot, maxEndNormalErr):
          minRadialDot     min over lateral faces of (outward face normal . radial unit vector);
                           must be close to +1 -- if it were the wrong face this would be ~0
          maxEndNormalErr  max deviation of the end-face normals from -x (low) and +x (high)
        Raises AssertionError if either check fails, so a wrong face ID cannot pass silently.
        """
        def face_centre_and_normal(el, fid):
            X = self.coords[[el[i] for i in self._FACE_NODES[fid]]]
            c = X.mean(axis=0)
            # outward normal of a planar-ish quad: diagonals cross product, oriented outward
            n = np.cross(X[2] - X[0], X[3] - X[1])
            nn = np.linalg.norm(n)
            return c, (n / nn if nn > 0 else n)

        minDot = np.inf
        for lab in self.lateral_elements():
            el = self.elements[lab - 1]
            c, n = face_centre_and_normal(el, self.FACE_LATERAL)
            r = np.array([0.0, c[1], c[2]])
            rn = np.linalg.norm(r)
            if rn < 1e-12:
                continue
            minDot = min(minDot, abs(float(n @ (r / rn))))
        err = 0.0
        for high, want in ((False, np.array([-1.0, 0, 0])), (True, np.array([1.0, 0, 0]))):
            fid = self.FACE_HIGH_X if high else self.FACE_LOW_X
            for lab in self.end_elements(high):
                el = self.elements[lab - 1]
                _, n = face_centre_and_normal(el, fid)
                err = max(err, float(np.linalg.norm(np.abs(n) - np.abs(want))))
        assert minDot > 0.99, f"FACE_LATERAL={self.FACE_LATERAL} is not the curved face " \
                              f"(min |n.e_r| = {minDot:.4f}); the face ID mapping is wrong"
        assert err < 1e-9, f"end-face normals are not axial (max err {err:.2e})"
        return minDot, err

    def cap_elements(self, capLen):
        """Elements within `capLen` of either end face.

        Implicit-gradient damage with natural (zero-flux) BCs reads HIGH at a boundary --
        the Helmholtz average cannot be diluted by material outside the specimen -- so
        damage nucleates at the loaded ends and saturates there before anything happens in
        the middle (measured: mid/end contrast 0.51, and strengthening the mid seed from
        x0.98 to x0.70 only moved it to 0.64).  Giving these elements a STRONGER card
        removes that artificial nucleation site, which is the standard "hard caps" recipe
        already used for the 3D shear-band example in this workspace.
        """
        c = self.centroids()
        return np.where((c[:, 0] < capLen) | (c[:, 0] > self.L - capLen))[0]

    def inp_blocks(self, elType, seed=None, cap=None):
        """the *node / *element / *nSet / *elSet part of an EdelweissFE input file"""
        out = ["*node"]
        out += [f"{i + 1}, {x:.10f}, {y:.10f}, {z:.10f}" for i, (x, y, z) in enumerate(self.coords)]
        out.append(f"*element, type={elType}, elSet=all")
        out += [f"{e + 1}, " + ", ".join(str(n + 1) for n in el) for e, el in enumerate(self.elements)]
        for name, idx in self.nsets().items():
            out.append(f"*nSet, nSet={name}")
            out += [", ".join(str(i + 1) for i in idx[k:k + 12]) for k in range(0, len(idx), 12)]
        if seed is not None and len(seed):
            out.append("*elSet, elSet=seed")
            out += [", ".join(str(i + 1) for i in seed[k:k + 12]) for k in range(0, len(seed), 12)]
        if cap is not None and len(cap):
            out.append("*elSet, elSet=caps")
            out += [", ".join(str(i + 1) for i in cap[k:k + 12]) for k in range(0, len(cap), 12)]
        # element sets + surfaces for the confining pressure and the end loads
        for nm, labs in (("lateralEls", self.lateral_elements()),
                         ("lowEndEls", self.end_elements(False)),
                         ("highEndEls", self.end_elements(True))):
            out.append(f"*elSet, elSet={nm}")
            out += [", ".join(str(v) for v in labs[k:k + 12]) for k in range(0, len(labs), 12)]
        for nm, es, fid in (("lateral", "lateralEls", self.FACE_LATERAL),
                            ("lowEnd", "lowEndEls", self.FACE_LOW_X),
                            ("highEnd", "highEndEls", self.FACE_HIGH_X)):
            out.append(f"*surface, type=element, name={nm}")
            out.append(f"{es}, {fid}")
        return "\n".join(out) + "\n"

    def midplane_polys(self, tol=1e-9):
        """(x, z) polygons of the elements having a face on the y = 0 plane, +z half.

        Both the +y and -y sides touch that plane; taking only z >= 0 ... no: we take the
        elements whose y=0 face exists AND whose centroid has y > 0, which tiles the half
        disc's mid-plane trace exactly once.  Returns (polys, elementIndices).
        """
        polys, idx = [], []
        for e, el in enumerate(self.elements):
            X = self.coords[el]
            on = np.abs(X[:, 1]) < tol
            if on.sum() != 4 or X[:, 1].mean() <= 0:
                continue
            P = X[on][:, [0, 2]]
            c = P.mean(axis=0)
            order = np.argsort(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
            polys.append(P[order])
            idx.append(e)
        return polys, np.array(idx)


if __name__ == "__main__":
    import sys

    R, L = 1.25, 5.0
    for nC, nR, nX in [(2, 1, 10), (2, 2, 10), (4, 2, 16)]:
        m = CylinderMesh(R=R, L=L, nC=nC, nR=nR, nX=nX)
        J = m.jacobians()
        V, Vex = m.volume(), math.pi * R * R * L
        polys, pidx = m.midplane_polys()
        print(f"nC={nC} nR={nR} nX={nX}: {len(m.coords):5d} nodes {len(m.elements):5d} els  "
              f"minJ={J.min():+.4e}  V={V:.4f} (exact {Vex:.4f}, {100*(V/Vex-1):+.2f}%)  "
              f"midplane cells={len(polys)}")
        assert J.min() > 0, "INVERTED ELEMENTS"
    print("\nnSets on the default mesh:")
    m = CylinderMesh(R=R, L=L)
    for k, v in m.nsets().items():
        print(f"  {k:6s} {len(v):4d} nodes")
    if "--write" in sys.argv:
        open("/tmp/cyl_blocks.inp", "w").write(m.inp_blocks("GC3D8UL"))
        print("wrote /tmp/cyl_blocks.inp")
