# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _
# | ____|__| | ___| |_      _____(_)___ ___
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \
# |_____\__,_|\___|_| \_/\_/_\___|_|___/___/
# |  \/  | ___  ___| |__  / _|_ __ ___  ___
# | |\/| |/ _ \/ __| '_ \| |_| '__/ _ \/ _ \
# | |  | |  __/\__ \ | | |  _| | |  __/  __/
# |_|  |_|\___||___/_| |_|_| |_|  \___|\___|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#
#  Research Group for Computational Mechanics of Materials
#  Institute of Structural Engineering, BOKU University, Vienna
#
#  2023 - today
#
#  Thomas Mader    |  thomas.mader@boku.ac.at
#
#  This file is part of EdelweissMeshfree.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissMeshfree.
#  ---------------------------------------------------------------------
"""
Triaxial compression of the Tournemire-shale HALF CYLINDER (Niandou et al. 1997; Mader, Schreter,
Hofstetter 2022, Sec. 4.3) with the finite-strain orthotropic damage-plasticity model
GRADIENTENHANCEDORTHOCDPFINITESTRAIN on smoothed-integration RKPM hexa particles.

This is the meshfree half of the FE/RKPM comparison of the paper's half-cylinder study.  The FE
half is Marmot/modules/materials/GradientEnhancedOrthoCDPFiniteStrain/testCases/edelweissFE/
niandou_triaxial_cyl.py, and everything that can be shared is shared:

  * the SAME discretisation: the butterfly (O-grid) hexa mesh of ``cyl_mesh.py`` (copied here
    from the FE test case), one ``GradientEnhancedFiniteStrainSQCNIxNSNI/3D/Hexa`` particle per
    finite element, its smoothing domain = the element; one RK kernel per particle at the
    particle centre, uniform support;
  * the SAME card (strict model of main.tex, calibrated Kelvin weights, convected frame, analytic
    tangents) and the SAME seed (fcy, fcu x 0.95 in a sphere of 0.25 R at the centre);
  * the SAME loading: step 1 the confining pressure on the curved surface and the top face, step 2
    the pressures held and a uniform axial displacement on the top face;
  * the SAME restraints, applied at the face VERTICES of the boundary particles by penalty:
    symmetry u_z = 0 on z = 0, u_x = 0 on the bottom face, u_y = 0 at one bottom particle near
    the axis.

Geometry: cylinder axis = x (0 .. 75 mm), radius 18.5 mm, half model z >= 0; bedding normal
n0 = (cos beta, sin beta, 0), i.e. in the symmetry plane.

Reported: the axial force carried IN ADDITION to the confinement is the penalty reaction of the
top platen, Delta F; t_dev(true) = Delta F / A(t) with the current top-face area, t_dev(nominal) =
Delta F / A0.

Usage:
    python ortho_finite_strain_half_cylinder_triaxial_test.py --beta 45 --h 5.0 --confine 30 \
        --umax 6.0 --tag HC_b45
"""

import argparse
import math
import os
import sys
import time

import edelweissfe.utils.performancetiming as performancetiming
import numpy as np
from edelweissfe.config.linsolve import getLinSolverByName
from edelweissfe.journal.journal import Journal
from edelweissfe.points.node import Node
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissfe.utils.exceptions import StepFailed

from edelweissmeshfree.constraints.particlepenaltyweakdirichtlet import (
    ParticlePenaltyWeakDirichlet,
)
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
    MarmotMeshfreeApproximationWrapper,
)
from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
    MarmotMeshfreeKernelFunctionWrapper,
)
from edelweissmeshfree.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.outputmanagers.ensight import (
    OutputManager as EnsightOutputManager,
)
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)
from edelweissmeshfree.sets.particleset import ParticleSet
from edelweissmeshfree.solvers.nqs import NonlinearQuasistaticSolver
from edelweissmeshfree.stepactions.particledistributedload import (
    ParticleDistributedLoad,
)

_EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLE_DIR)
from cyl_mesh import CylinderMesh  # noqa: E402

# =============================================================================================
#  card: Table 3 of Mader et al. (2022), strict model, calibrated weights (main.tex Table
#  fw:tab:niandoucard) -- identical to niandou_triaxial_cyl.py
# =============================================================================================
E1, E2 = 7000.0, 18000.0  # 1 = normal to the stratification planes
NU12, NU23 = 0.2, 0.25
G12 = 4000.0
G23 = E2 / (2.0 * (1.0 + NU23))
FCU, FCY = 42.54, 22.25
FTU, FBU = 9.1608, 43.8059
DF = 0.85
AH, BH, CH, DH = 0.022, 0.01, 1.0, 1e-6
AS, SOFTMOD, MAXDMG = 15.0, 4.75e-4, 0.9999
WEIGHTS = (1.0, 0.90, 0.90, 1.25, 1.25, 0.90)
L_NONLOCAL, WEIGHT_M = 5.0, 1.05
DAMAGE_ONSET, H_RESIDUAL = 1.0, 0.0

RADIUS, HEIGHT = 18.5, 75.0
A0_HALF = 0.5 * math.pi * RADIUS**2
SEED_SCALE = 0.95
SEED_RAD = 0.25 * RADIUS

# Abaqus C3D8 faces (0-based vertex indices), as in MarmotMeshfreeQuadHexCell.h
HEX_FACES = {1: (0, 3, 2, 1), 2: (4, 5, 6, 7), 3: (0, 1, 5, 4), 4: (1, 2, 6, 5), 5: (2, 3, 7, 6), 6: (3, 0, 4, 7)}


def card(beta, frameUpdate=1.0, fac=1.0, tangent=0.0, weights=WEIGHTS):
    b = math.radians(beta)
    return np.array(
        [
            E1, E2, E2, NU12, NU12, NU23, G12, G12, G23,
            math.cos(b), math.sin(b), 0.0,
            FCY * fac, FCU * fac, FBU, FTU, DF,
            AH, BH, CH, DH, AS, SOFTMOD, MAXDMG,
            *weights,
            L_NONLOCAL, WEIGHT_M, frameUpdate, DAMAGE_ONSET, H_RESIDUAL, tangent,
        ]
    )


def mesh(h):
    """The FE driver's O-grid: axial and outer-arc element sizes ~h."""
    nC = max(2, 2 * int(round(2.0 * math.pi * RADIUS / (4.0 * h) / 2.0)))
    nR = max(1, int(round(0.5 * RADIUS / h)))
    nX = max(2, int(round(HEIGHT / h)))
    return CylinderMesh(R=RADIUS, L=HEIGHT, nC=nC, nR=nR, nX=nX, coreFrac=0.5, half=True)


def boundaryFaces(m, tol=1e-7):
    """{name: {faceID: [element index]}} for the lateral surface, the top, the bottom and the
    symmetry plane, found geometrically from the four vertices of each face."""
    out = {"lateral": {}, "top": {}, "bottom": {}, "sym": {}}
    for e, el in enumerate(m.elements):
        X = m.coords[el]
        for fid, vs in HEX_FACES.items():
            F = X[list(vs)]
            if np.all(np.abs(np.hypot(F[:, 1], F[:, 2]) - RADIUS) < 1e-6):
                out["lateral"].setdefault(fid, []).append(e)
            elif np.all(np.abs(F[:, 0] - HEIGHT) < tol):
                out["top"].setdefault(fid, []).append(e)
            elif np.all(np.abs(F[:, 0]) < tol):
                out["bottom"].setdefault(fid, []).append(e)
            elif np.all(np.abs(F[:, 2]) < tol):
                out["sym"].setdefault(fid, []).append(e)
    return out


def faceArea(V):
    """Area of a (possibly warped) quad from its diagonals."""
    return 0.5 * np.linalg.norm(np.cross(V[2] - V[0], V[3] - V[1]))


def run_sim(beta=45.0, h=5.0, confine=30.0, umax=6.0, tag=None, ensight=True, frameUpdate=1.0,
            supportFactor=2.0, tangent=0.0, dduAbs=1e-4, fluxAbs=None, nlFluxAbs=1e-8, inc=0.01):
    np.set_printoptions(linewidth=200, precision=4)
    dimension = 3
    tag = tag or f"hc_b{int(beta)}_s{int(confine)}_h{h:g}"
    journal = Journal()
    theModel = MPMModel(dimension)

    m = mesh(h)
    els = m.elements
    nEl = len(els)
    faces = boundaryFaces(m)
    seed = set(m.seed_elements(mode="sphere", radius=SEED_RAD).tolist())
    edge = np.array(
        [max(np.linalg.norm(m.coords[el][a] - m.coords[el][b])
             for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (0, 4))) for el in els]
    )
    hRef = float(np.percentile(edge, 90))
    support = supportFactor * hRef
    journal.message(
        f"{tag}: beta = {beta} deg, t_conf = {confine} MPa, target h = {h} mm "
        f"(nC={m.nC} nR={m.nR} nX={m.nX}), {nEl} hexa particles, {len(seed)} seed, "
        f"edge 90 % = {hRef:.2f} mm, max {edge.max():.2f} mm, support {support:.2f} mm",
        "setup",
    )

    theApproximation = MarmotMeshfreeApproximationWrapper(
        "ReproducingKernelImplicitGradient", dimension, completenessOrder=1
    )
    cards = {
        f: {"material": "GRADIENTENHANCEDORTHOCDPFINITESTRAIN",
            "properties": card(beta, frameUpdate, f, tangent)}
        for f in (1.0, SEED_SCALE)
    }
    pName = "GradientEnhancedFiniteStrainSQCNIxNSNI/3D/Hexa"

    particles = []
    for e, el in enumerate(els):
        verts = m.coords[el].copy()
        centre = verts.mean(axis=0)
        number = e + 1
        p = MarmotParticleWrapper(pName, number, verts, 0.0, theApproximation,
                                  cards[SEED_SCALE if e in seed else 1.0])
        theModel.particles[number] = p
        particles.append(p)
        kf = MarmotMeshfreeKernelFunctionWrapper(
            Node(number, centre.copy()), "BSplineBoxed", supportRadius=support, continuityOrder=2
        )
        theModel.meshfreeKernelFunctions[number] = kf
        theModel.nodes[number] = kf.node
    theModel.particleSets["cyl_all"] = ParticleSet("cyl_all", particles)

    domain = ParticleKernelDomain(particles, list(theModel.meshfreeKernelFunctions.values()))
    theModel.particleKernelDomains["all_with_all"] = domain
    particleManager = KDBinOrganizedParticleManager(domain, dimension, journal,
                                                    bondParticlesToKernelFunctions=True)
    theModel.prepareYourself(journal)
    journal.printPrettyTable(theModel.makePrettyTableSummary(), "summary")


    fo = MPMFieldOutputController(theModel, journal)
    allSet = theModel.particleSets["all"]
    for name in ("displacement", "stress", "omega", "alphaP", "frameRotation", "materialAxis1",
                 "materialAxis2"):
        fo.addPerParticleFieldOutput(name, allSet, name)
    fo.addPerParticleFieldOutput("vertex displacements", allSet, "vertex displacements",
                                 f_x=lambda x: np.reshape(x, (-1, 3)))
    fo.initializeJob()

    outputManagers = []
    if ensight:
        ens = EnsightOutputManager(f"_ensight_{tag}", theModel, fo, journal, None)
        ens.createPerElementOutput(fo.fieldOutputs["displacement"])
        ens.createPerNodeOutput(fo.fieldOutputs["vertex displacements"])
        for name in ("omega", "alphaP", "frameRotation", "materialAxis1", "stress"):
            ens.createPerElementOutput(fo.fieldOutputs[name])
        ens.initializeJob()
        outputManagers.append(ens)

    # ------------------------------------------------------------------- restraints / loads
    PEN = 1e8

    def faceBCs(name, group, values):
        return [ParticlePenaltyWeakDirichlet(f"{name}{fid}", theModel, [particles[e] for e in es],
                                             "displacement", values, PEN, constrain=list(HEX_FACES[fid]))
                for fid, es in faces[group].items()]

    cent = np.array([m.coords[el].mean(axis=0) for el in els])
    bot = [e for es in faces["bottom"].values() for e in es]
    pin = [particles[min(bot, key=lambda e: np.hypot(cent[e, 1], cent[e, 2]))]]
    fixed = faceBCs("sym", "sym", {2: 0.0}) + faceBCs("bot", "bottom", {0: 0.0}) + [
        ParticlePenaltyWeakDirichlet("pinY", theModel, pin, "displacement", {1: 0.0}, PEN)
    ]

    loads = []
    if confine > 0.0:
        # one surface per (group, faceID): a face ID may occur in both groups
        for grp in ("lateral", "top"):
            for fid, es in faces[grp].items():
                sname = f"conf_{grp}_{fid}"
                theModel.surfaces[sname] = EntityBasedSurface(sname, {fid: [particles[e] for e in es]})
                loads.append(ParticleDistributedLoad(sname, theModel, journal, theModel.surfaces[sname],
                                                     "pressure", np.array([-confine])))

    iterationOptions = {
        "max. iterations": 40,
        "critical iterations": 10,
        "allowed residual growths": 15,
        "line search": True,
        "line search after n iterations": 6,
        "line search every n iterations": 2,
        "line search alphas": [0.25, 0.5, 0.75, 1.0],
    }
    if dduAbs:
        iterationOptions["spec. absolute field correction tolerances"] = {"displacement": dduAbs}
    if fluxAbs:
        iterationOptions.setdefault("spec. absolute flux residual tolerances", {})["displacement"] = fluxAbs
    if nlFluxAbs:
        iterationOptions.setdefault("spec. absolute flux residual tolerances", {})["nonlocal damage"] = nlFluxAbs
        iterationOptions.setdefault("spec. absolute field correction tolerances", {})["nonlocal damage"] = 1e-8
    linearSolver = getLinSolverByName("pardiso", {})
    solver = NonlinearQuasistaticSolver(journal)

    # ------------------------------------------------------------------- history
    topFaces = [(e, fid) for fid, es in faces["top"].items() for e in es]
    xc = cent[:, 0]
    midSlab = np.abs(xc - 0.5 * HEIGHT) <= 0.75 * h
    history = []
    snapshots = []
    ref = {"u0": None}
    topBCs = []

    def topArea(vd):
        return sum(faceArea(m.coords[els[e]][list(HEX_FACES[fid])] + vd[e][list(HEX_FACES[fid])])
                   for e, fid in topFaces)

    def record():
        f = fo.fieldOutputs
        vd = f["vertex displacements"].getLastResult().reshape(nEl, 8, 3)
        uTop = np.mean([vd[e][list(HEX_FACES[fid])][:, 0].mean() for e, fid in topFaces])
        if ref["u0"] is None:
            ref["u0"] = uTop
        tau = f["stress"].getLastResult().reshape(-1, 3, 3)
        omega = f["omega"].getLastResult().reshape(-1).copy()
        R = sum(c.penaltyForce[0] for c in topBCs) if topBCs else 0.0
        A = topArea(vd)
        history.append((-(uTop - ref["u0"]) / HEIGHT, R, A, float(omega.max()),
                        float(f["frameRotation"].getLastResult().max()),
                        float(tau[midSlab, 0, 0].mean()), float(tau[:, 1, 1].mean()), theModel.time))
        snapshots.append(dict(vd=vd.copy(), omega=omega,
                              alphaP=f["alphaP"].getLastResult().reshape(-1).copy(),
                              frameRotation=f["frameRotation"].getLastResult().reshape(-1).copy(),
                              axis1=f["materialAxis1"].getLastResult().reshape(-1, 3).copy()))
        h_ = history[-1]
        journal.message(f"  shortening {h_[0] * 100:6.3f} %  dF {2 * R / 1000:8.3f} kN (full)  "
                        f"t_dev true {R / A:7.2f}  nominal {R / A0_HALF:7.2f} MPa  omega_max {h_[3]:.4f}  "
                        f"R^p_max {h_[4]:.2f} deg", "record")

    class _Recorder:
        def initializeJob(self):
            pass

        def initializeStep(self, *a, **kw):
            pass

        def finalizeIncrement(self, *a, **kw):
            record()

        def finalizeFailedIncrement(self, *a, **kw):
            pass

        def finalizeStep(self, *a, **kw):
            pass

        def finalizeJob(self):
            pass

    t0 = time.time()
    if loads:
        journal.message(f"STEP 1 -- confining pressure {confine} MPa", "step")
        solver.solveStep(
            AdaptiveTimeStepper(theModel.time, 1.0, 0.25, 0.5, 1e-3, 100, journal),
            linearSolver, theModel, fo, outputManagers=outputManagers,
            particleManagers=[particleManager], constraints=fixed,
            particleDistributedLoads=loads, userIterationOptions=iterationOptions,
        )
        for dl in loads:
            dl.applyAtStepEnd(theModel)  # latch (see ex152): the meshfree NQS does not
        tau = fo.fieldOutputs["stress"].getLastResult().reshape(-1, 3, 3)
        journal.message(f"confinement latched: mean tau_xx {tau[:, 0, 0].mean():.3f}, tau_yy "
                        f"{tau[:, 1, 1].mean():.3f}, tau_zz {tau[:, 2, 2].mean():.3f} MPa", "step")

    nCover = np.array([len(p.kernelFunctions) for p in particles])
    journal.message(f"kernels per particle: min {nCover.min()}, mean {nCover.mean():.1f}, "
                    f"max {nCover.max()}", "setup")
    topBCs[:] = faceBCs("top", "top", {0: -abs(umax)})
    record()
    journal.message(f"STEP 2 -- axial shortening {umax} mm, confinement held", "step")
    stepFailed = False
    try:
        solver.solveStep(
            AdaptiveTimeStepper(theModel.time, 1.0, inc, 4.0 * inc, 1e-5, 2000, journal),
            linearSolver, theModel, fo, outputManagers=outputManagers + [_Recorder()],
            particleManagers=[particleManager], constraints=fixed + topBCs,
            particleDistributedLoads=loads, userIterationOptions=iterationOptions,
        )
    except StepFailed as e:
        journal.message(f"step stopped early: {e}", "warning")
        stepFailed = True

    fo.finalizeJob()
    for om in outputManagers:
        om.finalizeJob()
    prettytable = performancetiming.makePrettyTable()
    prettytable.min_table_width = journal.linewidth
    journal.printPrettyTable(prettytable, "Summary")

    hist = np.array(history)
    np.savez_compressed(
        os.path.join(_EXAMPLE_DIR, f"{tag}.npz"),
        history=hist, coords=m.coords, elements=np.array(els), cent=cent, seed=np.array(sorted(seed)),
        vd=np.array([s["vd"] for s in snapshots]), omega=np.array([s["omega"] for s in snapshots]),
        alphaP=np.array([s["alphaP"] for s in snapshots]),
        frameRotation=np.array([s["frameRotation"] for s in snapshots]),
        axis1=np.array([s["axis1"] for s in snapshots]),
        beta=beta, confine=confine, h=h, A0half=A0_HALF, stepFailed=stepFailed, wall=time.time() - t0,
    )
    tdev = hist[:, 1] / hist[:, 2]
    i = int(np.argmax(tdev))
    print(f"\n{tag}: reached {hist[-1, 0] * 100:.2f} % shortening in {len(hist)} records, "
          f"{time.time() - t0:.0f} s")
    print(f"  peak t_dev (true) {tdev[i]:.2f} MPa at {hist[i, 0] * 100:.2f} %, nominal "
          f"{(hist[:, 1] / A0_HALF).max():.2f} MPa; max omega {hist[:, 3].max():.4f}; "
          f"max R^p {hist[:, 4].max():.2f} deg")
    return hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=45.0)
    ap.add_argument("--h", type=float, default=5.0)
    ap.add_argument("--confine", type=float, default=30.0)
    ap.add_argument("--umax", type=float, default=6.0)
    ap.add_argument("--inc", type=float, default=0.01)
    ap.add_argument("--support", type=float, default=2.0)
    ap.add_argument("--frame", type=float, default=1.0)
    ap.add_argument("--tangent", type=float, default=0.0)
    ap.add_argument("--ddu-abs", type=float, default=1e-4)
    ap.add_argument("--flux-abs", type=float, default=None)
    ap.add_argument("--nl-flux-abs", type=float, default=1e-8)
    ap.add_argument("--no-ensight", action="store_true")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    run_sim(beta=a.beta, h=a.h, confine=a.confine, umax=a.umax, tag=a.tag, ensight=not a.no_ensight,
            frameUpdate=a.frame, supportFactor=a.support, tangent=a.tangent, dduAbs=a.ddu_abs,
            fluxAbs=a.flux_abs, nlFluxAbs=a.nl_flux_abs, inc=a.inc)
