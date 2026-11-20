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
"""Projection of a per-material-point value onto a B-spline space of LOWER degree than the cells, e.g. the B-bar /
F-bar projection of Elguedj, Bazilevs, Calo and Hughes (CMAME 197, 2008) for B-spline MPM cells.

The cells (Marmot ``hasMaterialPointProjection``, e.g. the ``DisplacementFBar/BSpline/<p>`` cells) report per material
point :math:`p` a value :math:`v_p(\\boldsymbol{Q})` (the F-bar cells: :math:`\\det\\Delta\\boldsymbol{F}_p`), a weight
:math:`w_p` (the volume at the start of the increment) and :math:`\\partial v_p/\\partial\\boldsymbol{Q}`, and receive a
projected value :math:`\\bar v_p` (F-bar: :math:`\\Delta\\bar{\\boldsymbol F}_p = (\\bar v_p/\\det\\Delta\\boldsymbol
F_p)^{1/n_\\mathrm{dim}}\\Delta\\boldsymbol F_p`, cell tangent at fixed :math:`\\bar v_p`, sensitivity
:math:`\\partial\\boldsymbol R/\\partial\\bar v_p`). This module projects the values onto the tensor-product B-spline
space :math:`\\bar{\\mathcal{S}}` of degree :math:`q = p - 1` on the SAME knot lines as the grid (open knot vectors,
:math:`C^{q-1} = C^{p-2}` across the knots), with the material points as quadrature points:

.. math::

    \\bar v_p = \\sum_A \\bar N_A(\\boldsymbol x_p)\\, \\theta_A , \\qquad \\boldsymbol\\theta = \\boldsymbol R\\,
    \\boldsymbol v .

Three projections :math:`\\boldsymbol R` (``mode``):

- ``"global"``: the L2 projection (Elguedj et al., eqs. of the B-bar/F-bar projection),
  :math:`\\sum_p w_p \\bar N_A (\\bar v_p - v_p) = 0` for all A, i.e. :math:`\\boldsymbol M\\boldsymbol\\theta =
  \\bar{\\boldsymbol N}^T\\boldsymbol W\\boldsymbol v`, :math:`\\boldsymbol M = \\bar{\\boldsymbol N}^T \\boldsymbol W
  \\bar{\\boldsymbol N}`. :math:`\\boldsymbol M^{-1}` is dense, so every value depends on every material point: the
  coupling is ONE dense block over all dofs of the projection (feasible up to ``maxDenseDofs``).
- ``"local"``: a local L2 projection: on every cell (or aggregate, see below) the values are projected onto the
  polynomials :math:`Q_q` of the cell (= the span of the :math:`(q+1)^d` functions of :math:`\\bar{\\mathcal S}`
  nonzero there), giving cell-wise coefficients :math:`c^e_A` of those functions; the global coefficient is their
  weighted mean :math:`\\theta_A = \\sum_e \\omega_{Ae} c^e_A`, :math:`\\omega_{Ae} = \\sum_{p\\in e}\\bar N_A w_p /
  \\sum_{p \\in \\mathrm{supp} A}\\bar N_A w_p` (Bezier-projection type averaging). The result lies in the continuous
  space :math:`\\bar{\\mathcal S}` (about one value per cell, as the global projection -- a cell-wise discontinuous
  :math:`Q_q` would impose :math:`p^d` constraints per cell and lock), reproduces :math:`Q_q` exactly (the local
  projections are exact for polynomials and the weights sum to one), and couples only cells within the support of a
  function. Needs :math:`\\geq (q+1)^d` material points per cell or aggregate (else the weighted mean there, i.e.
  only constants are reproduced; cond(Gram) > 1e12).
- ``"lumped"``: the row-sum lumped L2 projection, :math:`\\theta_A = \\sum_p \\bar N_A w_p v_p / \\sum_p \\bar N_A w_p`
  (the nodal F-bar on the degree-lowered space; reproduces constants only).

**Aggregation (AgMPM).** With an ``aggregation``
(:class:`~edelweissmeshfree.numerics.aggregation.CellAggregation`), the projection space is aggregated as the
displacements are, on the same aggregates: for ``"global"``/``"lumped"``, the "outer" functions of
:math:`\\bar{\\mathcal S}` (no interior cell in their support) are extended B-splines, their coefficient being that of
the polynomial piece of the connected interior cell, :math:`\\theta_b = \\sum_a E_{ba}\\theta_a` (local interpolation
as in :meth:`~edelweissmeshfree.numerics.aggregation.CellAggregation._bSplineExtensionTo`), i.e. the basis is
:math:`\\bar{\\boldsymbol N}\\boldsymbol E`; for ``"local"``, a boundary cell takes the polynomial of its aggregate
(projection over all material points of the interior cell and its boundary cells, with the polynomial pieces of the
interior cell). Without, small cuts carry their own (badly determined) projection values.

The coupling is exact, :math:`\\partial\\boldsymbol R_\\mathrm{int}/\\partial\\boldsymbol Q|_\\mathrm{coupling} =
\\sum_A \\boldsymbol L_A \\otimes \\boldsymbol H_A`, :math:`\\boldsymbol L_A = \\sum_p \\bar N_A(\\boldsymbol x_p)\\,
\\partial \\boldsymbol R/\\partial\\bar v_p`, :math:`\\boldsymbol H_A = \\partial\\theta_A/\\partial\\boldsymbol Q =
\\sum_p R_{Ap}\\,\\partial v_p/\\partial\\boldsymbol Q`, one entity per coefficient (``"local"``, ``"lumped"``) or one
dense entity (``"global"``), assembled before the multi-point constraints condense the system.

The projection is incremental and in the configuration at the start of the increment (as the F-bar cells): positions
and weights of the material points are those of the start of the increment, so :math:`\\boldsymbol R` is constant
during the Newton iterations of an increment. In the small-strain limit (one increment from the reference state) it is
the B-bar projection of Elguedj et al.; at finite strain it is the updated-Lagrangian incremental form of their F-bar
(the total :math:`\\bar J` is not projected, since the grid is reset every increment in MPM).

Use: ``AggregatedMaterialPointMethod(..., projectionMode="spline", splineProjection="global"|"local"|"lumped",
splineExtension=True)``, or without aggregation ``model.cellProjections["thetaBar"] = SplineProjection(...)``.
"""

import itertools

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparselinalg
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.numerics.aggregation import bSplineDegree, isBSplineCell


def bSplinePieces1D(knots: np.ndarray, degree: int, span: int, x: np.ndarray) -> np.ndarray:
    """The ``degree + 1`` B-splines nonzero on the knot span ``knots[span] <= x < knots[span + 1]``, as polynomials
    (evaluated also outside the span: the polynomial pieces of that span, extended), Cox-de Boor (Piegl and Tiller,
    A2.2), vectorised over the points.

    Returns
    -------
    np.ndarray
        ``(len(x), degree + 1)``, column k = B-spline ``span - degree + k``.
    """

    x = np.atleast_1d(np.asarray(x, dtype=float))
    N = np.zeros((x.size, degree + 1))
    N[:, 0] = 1.0
    left = np.zeros((x.size, degree + 1))
    right = np.zeros((x.size, degree + 1))
    for j in range(1, degree + 1):
        left[:, j] = x - knots[span + 1 - j]
        right[:, j] = knots[span + j] - x
        saved = np.zeros(x.size)
        for r in range(j):
            temp = N[:, r] / (right[:, r + 1] + left[:, j - r])
            N[:, r] = saved + right[:, r + 1] * temp
            saved = left[:, j - r] * temp
        N[:, j] = saved
    return N


class TensorBSplineSpace:
    """Tensor-product B-splines of a given degree on open knot vectors over given knot lines.

    Parameters
    ----------
    lines
        Per direction, the (strictly increasing) knot lines, i.e. the cell boundaries of the grid.
    degree
        The degree q >= 0 (C^{q-1} across the interior knots).
    """

    def __init__(self, lines: list, degree: int):
        self.dimension = len(lines)
        self.degree = int(degree)
        self.lines = [np.asarray(line, dtype=float) for line in lines]
        q = self.degree
        self.knots = [np.r_[[line[0]] * q, line, [line[-1]] * q] for line in self.lines]
        self.nSpans = [len(line) - 1 for line in self.lines]
        self.nFunctions = [n + q for n in self.nSpans]
        self._offsets = np.array(list(itertools.product(range(q + 1), repeat=self.dimension)), dtype=int)

    @property
    def nLocal(self) -> int:
        """The number of functions nonzero on a span, (q+1)^d."""
        return (self.degree + 1) ** self.dimension

    def spanOf(self, boxMin, boxMax) -> tuple:
        """The span (multi-index) of a cell given by its bounding box (= its knot span)."""
        centre = 0.5 * (np.asarray(boxMin, dtype=float) + np.asarray(boxMax, dtype=float))
        return tuple(
            int(np.clip(np.searchsorted(self.lines[d], centre[d], side="right") - 1, 0, self.nSpans[d] - 1))
            for d in range(self.dimension)
        )

    def functionsOn(self, span: tuple) -> np.ndarray:
        """The flat indices of the (q+1)^d functions nonzero on a span, in the order of :meth:`evaluate`."""
        multi = np.asarray(span, dtype=int)[None, :] + self._offsets
        return np.ravel_multi_index(tuple(multi.T), self.nFunctions)

    def multiIndex(self, function: int) -> tuple:
        return np.unravel_index(int(function), self.nFunctions)

    def supportSpans(self, function: int) -> list:
        """The spans (multi-indices) of the support of a function."""
        i = self.multiIndex(function)
        ranges = [range(max(0, i[d] - self.degree), min(self.nSpans[d] - 1, i[d]) + 1) for d in range(self.dimension)]
        return list(itertools.product(*ranges))

    def grevillePoint(self, function: int) -> np.ndarray:
        i = self.multiIndex(function)
        q = self.degree
        if q == 0:
            return np.array([0.5 * (self.lines[d][i[d]] + self.lines[d][i[d] + 1]) for d in range(self.dimension)])
        return np.array([self.knots[d][i[d] + 1 : i[d] + q + 1].mean() for d in range(self.dimension)])

    def evaluate(self, span: tuple, points: np.ndarray) -> np.ndarray:
        """The polynomial pieces of the functions nonzero on ``span`` at points (also outside the span: extension).

        Returns
        -------
        np.ndarray
            ``(nPoints, (q+1)^d)``, columns in the order of :meth:`functionsOn`.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        q = self.degree
        values = np.ones((points.shape[0], self.nLocal))
        for d in range(self.dimension):
            pieces = bSplinePieces1D(self.knots[d], q, span[d] + q, points[:, d])
            values *= pieces[:, self._offsets[:, d]]
        return values

    def chebyshevPoints(self, span: tuple) -> np.ndarray:
        """(q+1)^d tensor-product Chebyshev points of a span (unisolvent for Q_q)."""
        n = self.degree + 1
        t = 0.5 * (1.0 - np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n)))
        axes = [self.lines[d][span[d]] + t * (self.lines[d][span[d] + 1] - self.lines[d][span[d]]) for d in range(
            self.dimension)]  # fmt: skip
        return np.array(np.meshgrid(*axes, indexing="ij")).reshape(self.dimension, -1).T

    def transfer(self, target: tuple, source: tuple) -> np.ndarray:
        """The coefficients, in the functions nonzero on span ``target``, of the polynomial pieces of the functions of
        span ``source`` (extended): ``T[k, a]`` with :math:`\\phi^{source}_a = \\sum_k T_{ka} \\phi^{target}_k` on the
        target span (local interpolation at Chebyshev points)."""
        if tuple(target) == tuple(source):
            return np.eye(self.nLocal)
        y = self.chebyshevPoints(target)
        return np.linalg.solve(self.evaluate(target, y), self.evaluate(source, y))


def spaceOfGrid(cells, dimension: int, degreeDrop: int = 1) -> TensorBSplineSpace:
    """The B-spline space of degree p - degreeDrop on the knot lines of a B-spline grid (its cells' knot spans)."""

    cells = list(cells)
    if not cells or not all(isBSplineCell(c) for c in cells):
        raise NotImplementedError("SplineProjection needs a grid of B-spline cells.")
    p = bSplineDegree(cells[0], dimension)
    q = p - degreeDrop
    if q < 0:
        raise ValueError("degree p - degreeDrop = {:} < 0".format(q))
    boxes = np.array([np.r_[c.getBoundingBox()[0], c.getBoundingBox()[1]] for c in cells])
    size = float(np.min(boxes[:, dimension:] - boxes[:, :dimension]))
    lines = []
    for d in range(dimension):
        values = np.r_[boxes[:, d], boxes[:, dimension + d]]
        values = np.sort(values)
        keep = np.r_[True, np.diff(values) > 1e-6 * size]
        lines.append(values[keep])
    return TensorBSplineSpace(lines, q)


class SplineProjectionCoupling(ConstraintBase):
    """The coupling through one coefficient (or, ``"global"``, all coefficients) of the projection: an entity without
    residual that adds :math:`\\boldsymbol L_A\\otimes\\boldsymbol H_A` to the tangent."""

    projectsOverCells = True

    def __init__(self, name: str, projection, index: int, nodes: list, fields: list, nDof: int):
        self._name = name
        self.projector = projection
        self._index = index
        self._nodes = nodes
        self._fields = fields
        self._nDof = nDof
        self.scalarVariables = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fields(self) -> list:
        return self._fields

    @property
    def nDof(self) -> int:
        return self._nDof

    def computeKernels(self, U_np: np.ndarray, dU: np.ndarray, P: np.ndarray, K: np.ndarray, timeStep: TimeStep):
        self.projector.addCoupling(self._index, K)


class SplineProjection:
    """Projection of per-material-point values onto the B-spline space of degree p - ``degreeDrop`` of the grid.

    Parameters
    ----------
    name
        The name.
    model
        The MPM model.
    journal
        The journal.
    mode
        ``"global"``, ``"local"`` or ``"lumped"`` (see the module).
    degreeDrop
        The projection space has degree p - degreeDrop (default 1, Elguedj et al.).
    maxDenseDofs
        ``"global"``: refuse a dense coupling block larger than this.
    """

    def __init__(
        self,
        name: str,
        model,
        journal: Journal = None,
        mode: str = "global",
        degreeDrop: int = 1,
        maxDenseDofs: int = 16000,
    ):
        if mode not in ("global", "local", "lumped"):
            raise ValueError("mode must be 'global', 'local' or 'lumped', not {:}".format(mode))
        self._name = name
        self._model = model
        self._journal = journal
        self.mode = mode
        self.degreeDrop = degreeDrop
        self.maxDenseDofs = maxDenseDofs
        self.space = None
        self._key = None
        self._entities = []
        self._cells = []
        self._evaluation = 0
        self._couplingEvaluation = -1
        self._increment = None
        #: coefficients theta of the last evaluation (independent ones), of the deviation v - 1
        self.coefficients = np.zeros(0)

    @property
    def name(self) -> str:
        return self._name

    def _message(self, message: str):
        if self._journal is not None:
            self._journal.message(message, self._name)

    # ------------------------------------------------------------------ structure (per set of active cells)

    def updateConnectivity(self, activeCells: list = None, aggregation=None):
        """Follow the active cells (once per increment, after the managers assigned the material points).

        Parameters
        ----------
        activeCells
            The active cells; default those of the model's MPM managers.
        aggregation
            Optional :class:`~edelweissmeshfree.numerics.aggregation.CellAggregation` of these active cells: aggregate
            the projection space on its aggregates (see the module).
        """

        self._increment = None  # positions and weights change every increment
        if activeCells is None:
            activeCells = self._model.activeCellsOf(self._model.mpmManagers)
        cells = [c for c in activeCells if getattr(c, "hasMaterialPointProjection", False)]
        key = (tuple(c.cellNumber for c in cells), None if aggregation is None else tuple(aggregation.interiorCells))
        if key == self._key:
            return
        self._key = key
        if self.space is None:
            self.space = spaceOfGrid(self._model.cells.values(), self._model.domainSize, self.degreeDrop)
        self._build(cells, aggregation)

    def _build(self, cells: list, aggregation):
        space = self.space
        dimension = self._model.domainSize
        self._cells = cells
        nC = len(cells)
        spans = [space.spanOf(*c.getBoundingBox()) for c in cells]
        functions = [space.functionsOn(s) for s in spans]
        self._spans = spans
        index = {c.cellNumber: i for i, c in enumerate(cells)}
        active = sorted({int(f) for fs in functions for f in fs})
        cellsOfFunction = {f: [] for f in active}
        for i, fs in enumerate(functions):
            for k, f in enumerate(fs):
                cellsOfFunction[int(f)].append((i, k))

        connected = {}
        interior = set()
        if aggregation is not None:
            interior = {index[c] for c in aggregation.interiorCells if c in index}
            connected = {
                index[b]: index[i] for b, i in aggregation.connectedInteriorCell.items() if b in index and i in index
            }
        self._aggregated = aggregation is not None
        nOuter = 0

        if self.mode in ("global", "lumped"):
            # independent functions; outer ones (no interior cell in the support) extended from an interior cell
            extension = {}
            if aggregation is not None and interior:
                inner = {int(f) for i in interior for f in functions[i]}
                boxes = [c.getBoundingBox() for c in cells]
                centres = {i: 0.5 * (np.asarray(b[0]) + np.asarray(b[1])) for i, b in enumerate(boxes)}
                for f in active:
                    if f in inner:
                        continue
                    holders = [i for i, _ in cellsOfFunction[f]]
                    candidates = sorted({connected[i] for i in holders if i in connected})
                    if not candidates:
                        continue  # a cell outside every aggregate: keep the function independent
                    x = space.grevillePoint(f)
                    dist = [np.linalg.norm(centres[c] - x) for c in candidates]
                    interiorCell = candidates[int(np.argmin(dist))]
                    dist = [np.linalg.norm(centres[c] - centres[interiorCell]) for c in holders]
                    reference = holders[int(np.argmin(dist))]
                    T = space.transfer(spans[reference], spans[interiorCell])
                    row = T[list(functions[reference]).index(f)]
                    big = 1e-13 * np.abs(row).max()
                    extension[f] = [(int(m), float(w)) for m, w in zip(functions[interiorCell], row) if abs(w) > big]
                nOuter = len(extension)
            independent = [f for f in active if f not in extension]
            position = {f: k for k, f in enumerate(independent)}
            cellIdx, cellMaps = [], []
            for i in range(nC):
                columns, entries = {}, []
                for k, f in enumerate(functions[i]):
                    for m, w in extension.get(int(f), [(int(f), 1.0)]):
                        col = columns.setdefault(position[m], len(columns))
                        entries.append((k, col, w))
                T = np.zeros((space.nLocal, len(columns)))
                for k, col, w in entries:
                    T[k, col] += w
                cellIdx.append(np.array(list(columns), dtype=int))
                cellMaps.append(T)
            nTheta = len(independent)
            self._groups = None
        else:  # local: all active functions are coefficients; groups = aggregates
            position = {f: k for k, f in enumerate(active)}
            cellIdx = [np.array([position[int(f)] for f in fs], dtype=int) for fs in functions]
            cellMaps = [np.eye(space.nLocal) for _ in range(nC)]
            nTheta = len(active)
            base = [connected.get(i, i) for i in range(nC)]
            members = {}
            for i in range(nC):
                members.setdefault(base[i], []).append(i)
            for b in members:  # the base cell first
                members[b].sort(key=lambda i: (i != b, i))
            self._groups = members
            self._baseOf = base
            self._transfers = [space.transfer(spans[i], spans[base[i]]) for i in range(nC)]
        self._cellIdx = cellIdx
        self._cellMaps = cellMaps
        self._nTheta = nTheta

        # node-wise dof layout of the cells
        nodeFields = {}
        for cell in cells:
            for node, fields in zip(cell.nodes, cell.fields):
                nodeFields.setdefault(node, list(fields))
        self._nodeFields = nodeFields

        def layout(entityCells):
            entityNodes = sorted({n for c in entityCells for n in cells[c].nodes}, key=lambda n: n.label)
            offset, n = {}, 0
            for node in entityNodes:
                offset[node] = n
                n += sum(getFieldSize(f, dimension) for f in nodeFields[node])
            dofs = {}
            for c in entityCells:
                cell = cells[c]
                nodeWise = []
                for node in cell.nodes:
                    nodeWise.extend(offset[node] + np.arange(sum(getFieldSize(f, dimension) for f in nodeFields[node])))
                nodeWise = np.asarray(nodeWise, dtype=int)
                perm = cell.dofIndicesPermutation
                dofs[c] = nodeWise[np.asarray(perm, dtype=int)] if perm is not None else nodeWise
            return entityNodes, [nodeFields[node] for node in entityNodes], n, dofs

        self._entities = []
        self._entityParts = []
        if self.mode == "global":
            entityNodes, entityFields, n, dofs = layout(list(range(nC)))
            if n > self.maxDenseDofs:
                raise RuntimeError(
                    "SplineProjection 'global': dense coupling over {:} dofs > maxDenseDofs = {:}; use mode='local'."
                    .format(n, self.maxDenseDofs)
                )  # fmt: skip
            self._entityParts.append(dofs)
            self._entities.append(
                SplineProjectionCoupling(self._name + " global", self, 0, entityNodes, entityFields, n)
            )
        else:
            lParts = [[] for _ in range(nTheta)]
            for c in range(nC):
                for k, m in enumerate(cellIdx[c]):
                    lParts[m].append((c, k))
            for m in range(nTheta):
                lCells = sorted({c for c, _ in lParts[m]})
                if self.mode == "lumped":
                    hCells = lCells
                    hParts = lParts[m]
                else:
                    hParts = [(e, k, cp) for e, k in lParts[m] for cp in self._groups[self._baseOf[e]]]
                    hCells = sorted({cp for _, _, cp in hParts})
                entityCells = sorted(set(lCells) | set(hCells))
                entityNodes, entityFields, n, dofs = layout(entityCells)
                self._entityParts.append((lParts[m], hParts, dofs))
                self._entities.append(
                    SplineProjectionCoupling("{:} {:}".format(self._name, m), self, m, entityNodes, entityFields, n)
                )
        self._message(
            "spline projection ({:}, degree {:}) over {:} cells: {:} coefficients ({:} outer functions extended){:}"
            .format(self.mode, space.degree, nC, nTheta, nOuter,
                    ", {:} aggregates".format(sum(len(g) > 1 for g in self._groups.values())) if self._groups else "")
        )  # fmt: skip

    def couplingEntities(self) -> list:
        return list(self._entities)

    # ------------------------------------------------------------------ per increment (positions, weights)

    def _prepareIncrement(self, weights: list):
        space = self.space
        cells = self._cells
        self._N = []
        positions = []
        for c, cell in enumerate(cells):
            X = np.array([np.asarray(mp.getCenterCoordinates(), dtype=float).ravel()[: space.dimension]
                          for mp in cell.assignedMaterialPoints])  # fmt: skip
            positions.append(X)
            self._N.append(space.evaluate(self._spans[c], X) @ self._cellMaps[c])
        nTheta = self._nTheta
        if self.mode == "lumped":
            D = np.zeros(nTheta)
            for c in range(len(cells)):
                np.add.at(D, self._cellIdx[c], (self._N[c] * weights[c][:, None]).sum(axis=0))
            self._D = D
        elif self.mode == "global":
            rows, cols, vals = [], [], []
            for c in range(len(cells)):
                Nc = self._N[c]
                Mc = (Nc * weights[c][:, None]).T @ Nc
                idx = self._cellIdx[c]
                rows.append(np.repeat(idx, len(idx)))
                cols.append(np.tile(idx, len(idx)))
                vals.append(Mc.ravel())
            M = sparse.csc_matrix(
                (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nTheta, nTheta)
            )
            diag = M.diagonal()
            empty = diag <= 1e-300
            if empty.any():
                M = M + sparse.diags(empty.astype(float), format="csc")
            self._M = M
            self._Mlu = sparselinalg.splu(M)
        else:  # local
            num = [(self._N[c] * weights[c][:, None]).sum(axis=0) for c in range(len(cells))]
            den = np.zeros(nTheta)
            for c in range(len(cells)):
                np.add.at(den, self._cellIdx[c], num[c])
            den[den <= 0.0] = 1.0
            self._R = []
            self._rankDeficient = 0
            Q = {}
            for b, group in self._groups.items():
                Phi = np.vstack([space.evaluate(self._spans[b], positions[c]) for c in group])
                w = np.concatenate([weights[c] for c in group])
                Gram = (Phi * w[:, None]).T @ Phi
                try:
                    cond = np.linalg.cond(Gram)
                except np.linalg.LinAlgError:
                    cond = np.inf
                if cond < 1e12:
                    Q[b] = np.linalg.solve(Gram, (Phi * w[:, None]).T)
                else:
                    # too few material points for Q_q: the weighted mean (constants are reproduced, every
                    # coefficient of the cell takes it, the basis being a partition of unity)
                    self._rankDeficient += 1
                    Q[b] = np.tile(w / w.sum(), (space.nLocal, 1))
            for e in range(len(cells)):
                omega = num[e] / den[self._cellIdx[e]]
                self._R.append((omega[:, None] * self._transfers[e]) @ Q[self._baseOf[e]])
            if self._rankDeficient:
                self._message("{:} cells/aggregates with too few material points for Q_{:}: weighted mean instead"
                              .format(self._rankDeficient, space.degree))  # fmt: skip
            # slices of the group's material points per member cell
            self._groupSlices = {}
            for b, group in self._groups.items():
                start = 0
                for c in group:
                    n = len(positions[c])
                    self._groupSlices[(b, c)] = slice(start, start + n)
                    start += n
        self._increment = True

    # ------------------------------------------------------------------ per evaluation

    def projectToCells(self, dU):
        """The projected values at the trial increment, set on every cell before the cells interpolate."""

        cells = self._cells
        values, weights, self._G = [], [], []
        for cell in cells:
            nP = len(cell.assignedMaterialPoints)
            v, w, dv = np.zeros(nP), np.zeros(nP), np.zeros(nP * cell.nDof)
            cell.computeMaterialPointProjectionContributions(np.ascontiguousarray(dU[cell]), v, w, dv)
            values.append(v)
            weights.append(w)
            self._G.append(dv.reshape(nP, cell.nDof))
        self._w = weights
        if self._increment is None:
            self._prepareIncrement(weights)
        # project the deviation from 1 (all projections reproduce constants): identical in exact arithmetic, but the
        # round-off then scales with v - 1, not with v; otherwise the bulk modulus amplifies it (residual floor of
        # 1e-6 at a zero increment, nu = 0.4999)
        values = [v - 1.0 for v in values]
        nTheta = self._nTheta
        theta = np.zeros(nTheta)
        if self.mode == "lumped":
            for c in range(len(cells)):
                np.add.at(theta, self._cellIdx[c], self._N[c].T @ (weights[c] * values[c]))
            positive = self._D > 0
            theta[positive] /= self._D[positive]
            theta[~positive] = 0.0
        elif self.mode == "global":
            b = np.zeros(nTheta)
            for c in range(len(cells)):
                np.add.at(b, self._cellIdx[c], self._N[c].T @ (weights[c] * values[c]))
            theta = self._Mlu.solve(b)
        else:
            for e in range(len(cells)):
                group = self._groups[self._baseOf[e]]
                vg = np.concatenate([values[c] for c in group])
                np.add.at(theta, self._cellIdx[e], self._R[e] @ vg)
        self.coefficients = theta
        for c, cell in enumerate(cells):
            cell.setMaterialPointProjectedValues(np.ascontiguousarray(1.0 + self._N[c] @ theta[self._cellIdx[c]]))
        self._evaluation += 1

    def _updateSensitivities(self):
        cells = self._cells
        self._L = []
        for c, cell in enumerate(cells):
            nP = self._N[c].shape[0]
            S = np.zeros(nP * cell.nDof)
            cell.getMaterialPointProjectionSensitivities(S)
            self._L.append(self._N[c].T @ S.reshape(nP, cell.nDof))
        if self.mode == "lumped":
            self._H = [((self._N[c] * self._w[c][:, None]).T @ self._G[c]) / self._D[self._cellIdx[c]][:, None]
                       for c in range(len(cells))]  # fmt: skip
        elif self.mode == "local":
            self._H = {}
            for e in range(len(cells)):
                b = self._baseOf[e]
                for cp in self._groups[b]:
                    self._H[(e, cp)] = self._R[e][:, self._groupSlices[(b, cp)]] @ self._G[cp]
        self._couplingEvaluation = self._evaluation

    def addCoupling(self, m: int, K: np.ndarray):
        """Add the coupling of entity ``m`` to its tangent block ``K`` (nDof x nDof)."""

        if self._couplingEvaluation != self._evaluation:
            self._updateSensitivities()
        entity = self._entities[m]
        nDof = entity.nDof
        if self.mode == "global":
            dofs = self._entityParts[0]
            nTheta = self._nTheta
            L = np.zeros((nDof, nTheta))
            B = np.zeros((nTheta, nDof))
            for c in range(len(self._cells)):
                idx, d = self._cellIdx[c], dofs[c]
                L[np.ix_(d, idx)] += self._L[c].T
                B[np.ix_(idx, d)] += (self._N[c] * self._w[c][:, None]).T @ self._G[c]
            H = self._Mlu.solve(B)
            K += L @ H
            return
        lParts, hParts, dofs = self._entityParts[m]
        L = np.zeros(nDof)
        H = np.zeros(nDof)
        for c, k in lParts:
            np.add.at(L, dofs[c], self._L[c][k])
        if self.mode == "lumped":
            for c, k in hParts:
                np.add.at(H, dofs[c], self._H[c][k])
        else:
            for e, k, cp in hParts:
                np.add.at(H, dofs[cp], self._H[(e, cp)][k])
        K += np.outer(L, H)
