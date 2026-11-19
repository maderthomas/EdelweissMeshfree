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
#  Matthias Neuner |  matthias.neuner@boku.ac.at
#  Thomas Mader    |  thomas.mader@bokut.ac.at
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
"""The fictitious mass of the dynamic relaxation from the estimated wave speed (Sutcliffe et al., IJNME 2026, Sec.
3.2.1, eq. 35), for EdelweissFE's :class:`~edelweissfe.solvers.dynamicrelaxation.DynamicRelaxation` with
``fictitiousMass=provider``:

.. math::

    \\tilde{M}_{vv} = \\zeta_M \\frac{\\Delta\\mathcal{T}^2}{4} \\sum_p \\left(\\frac{\\mathcal{U}_\\mathrm{est}}{h}
    \\right)^2 M_p \\, V_p ,

summed over the material points :math:`p` whose cell has the node :math:`v`, with :math:`M_p` the P-wave modulus of
the material point, :math:`h` the cell size, :math:`V_p = V_p^0 \\det \\boldsymbol{F}` and the geometric factor
:math:`\\mathcal{U}_\\mathrm{est} = \\max([1\\;1\\;1] \\Delta\\boldsymbol{F}^{-1})` (eq. 36; here the largest absolute
column sum of :math:`\\Delta\\boldsymbol{F}^{-1}` over the ``nDim`` directions of the model: a rotation may make a
column sum negative). With ``henckyEnhancement`` the modulus is :math:`M_p' = \\exp(-\\varepsilon_{III}) M_p / \\det
\\boldsymbol{F}` (eq. 38, :math:`\\varepsilon_{III}` the smallest principal logarithmic strain of
:math:`\\boldsymbol{F} = \\Delta\\boldsymbol{F} \\boldsymbol{F}_n`).

The value of a material point enters every DOF of every node of its cell alike, so the estimator is independent of
the fields and their layout; only the modulus is physics: ``pWaveModulus`` is a number (e.g. :math:`K + \\tfrac{4}{3}
G`, eq. 50) or a function of a list of material points returning their moduli (e.g. :math:`K` for a yielding point of
perfect von Mises plasticity, eq. 51). The material points need ``"delta deformation gradient"`` and ``"deformation gradient"`` results, and their
volume at creation (``initialVolume``) unless ``initialVolume`` is given.

The paper chooses :math:`\\zeta_M = 2` (``stabilityFactor``), and notes that :math:`\\zeta_M = 1` generally converges
with aggregation.

.. code-block:: python

    from edelweissfe.solvers.dynamicrelaxation import DynamicRelaxation
    from edelweissmeshfree.solvers.pwavemass import PWaveFictitiousMass

    mass = PWaveFictitiousMass(mpmModel, pWaveModulus=K + 4.0 / 3.0 * G)
    solver = DynamicRelaxation(jobInfo, journal, fictitiousMass="provider", fictitiousMassProvider=mass)
"""

from collections.abc import Callable

import numpy as np


class PWaveFictitiousMass:
    """The P-wave fictitious mass of eq. 35 (see the module documentation).

    Parameters
    ----------
    model
        The MPM model.
    pWaveModulus
        The small-strain P-wave modulus of the material points: a number, or a function of a list of material points
        returning their moduli (an array).
    stabilityFactor
        :math:`\\zeta_M` (paper: 2).
    henckyEnhancement
        Apply the enhancement of eq. 38.
    initialVolume
        A function of the material point giving its volume at creation; default: its ``initialVolume``.
    """

    def __init__(
        self,
        model,
        pWaveModulus: float | Callable,
        stabilityFactor: float = 2.0,
        henckyEnhancement: bool = True,
        initialVolume: Callable | None = None,
    ):
        self.model = model
        self.pWaveModulus = pWaveModulus
        self.stabilityFactor = stabilityFactor
        self.henckyEnhancement = henckyEnhancement
        self.initialVolume = initialVolume if initialVolume is not None else (lambda mp: mp.initialVolume)
        self.nDim = model.domainSize
        self._views = {}

    def _viewsOf(self, mp):
        """The persistent views of the deformation gradients of a material point, and its initial volume."""
        v = self._views.get(id(mp))
        if v is None or v[0] is not mp:
            v = (
                mp,
                mp.getResultArray("delta deformation gradient"),
                mp.getResultArray("deformation gradient"),
                float(self.initialVolume(mp)),
            )
            self._views[id(mp)] = v
        return v

    def materialPointTerms(self, mps: list) -> np.ndarray:
        """:math:`(\\mathcal{U}_\\mathrm{est})^2 M_p' V_p` of the material points (without :math:`1/h^2`)."""

        nDim = self.nDim
        views = [self._viewsOf(mp) for mp in mps]
        dF = np.array([v[1][:9] for v in views], dtype=float).reshape(-1, 3, 3)
        Fn = np.array([v[2][:9] for v in views], dtype=float).reshape(-1, 3, 3)
        V0 = np.array([v[3] for v in views])
        F = dF @ Fn
        J = np.linalg.det(F)
        U = np.abs(np.linalg.inv(dF[:, :nDim, :nDim]).sum(axis=1)).max(axis=1)
        if callable(self.pWaveModulus):
            M = np.asarray(self.pWaveModulus(mps), dtype=float)
        else:
            M = np.full(len(mps), float(self.pWaveModulus))
        if self.henckyEnhancement:
            epsMin = 0.5 * np.log(np.linalg.eigvalsh(F @ np.transpose(F, (0, 2, 1))).min(axis=1))
            M = M * np.exp(-epsMin) / J
        return U * U * M * V0 * J

    @staticmethod
    def cellSize(cell) -> float:
        """The size :math:`h` of a cell: the smallest extent of its bounding box."""
        lo, hi = cell.getBoundingBox()
        return float(np.min(np.asarray(hi) - np.asarray(lo)))

    def fictitiousMass(self, solver, pseudoTimeStep: float) -> np.ndarray:
        """The diagonal fictitious mass of the cells of the discrete system the solver solves, a full-size vector;
        zero at DOFs of no cell (the solver adds the Gerschgorin row sums of the other entities).

        Parameters
        ----------
        solver
            The solver, whose ``system`` (``locations``, ``elements``) is the discrete system being solved, and whose
            ``_U`` is a vector of it.
        pseudoTimeStep
            The fictitious time step.

        Returns
        -------
        np.ndarray
            The diagonal of the fictitious mass.
        """

        system = solver.system
        locations = system.locations
        cells, mps, owner = [], [], []
        for cell in system.elements:
            cellMPs = getattr(cell, "assignedMaterialPoints", None)
            if not cellMPs:
                continue
            owner += [len(cells)] * len(cellMPs)
            mps += cellMPs
            cells.append(cell)
        m = np.zeros(len(solver._U))
        if not cells:
            return m
        h = np.array([self.cellSize(cell) for cell in cells])
        perCell = np.bincount(np.array(owner), weights=self.materialPointTerms(mps), minlength=len(cells)) / h**2
        scale = self.stabilityFactor * pseudoTimeStep**2 / 4.0
        for cell, s in zip(cells, perCell):
            np.add.at(m, locations[cell], scale * s)
        return m
