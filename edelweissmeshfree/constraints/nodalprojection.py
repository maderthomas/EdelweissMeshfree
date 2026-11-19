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
"""A projection with one value per material point through the nodes of the background grid, e.g. the nodal F-bar
(smoothed volume change) for implicit MPM.

Every cell that supports it (``hasMaterialPointProjection``) reports per material point :math:`p` a value
:math:`v_p(\\boldsymbol{Q})`, a weight :math:`w_p` and the cell basis :math:`N_a(\\boldsymbol{x}_p)` it uses. The values
are projected onto the nodes of the grid by the lumped (row-sum) L2 projection and interpolated back,

.. math::

    v_a = \\frac{\\sum_p N_a(\\boldsymbol{x}_p)\\, w_p\\, v_p}{\\sum_p N_a(\\boldsymbol{x}_p)\\, w_p}, \\qquad
    \\bar{v}_p = \\sum_a N_a(\\boldsymbol{x}_p)\\, v_a ,

over all material points of all active cells, and every cell receives its :math:`\\bar{v}_p`. For the nodal F-bar of
the Marmot ``DisplacementFBar`` cells, :math:`v_p = \\det\\Delta\\boldsymbol{F}_p` and :math:`w_p = V_p` (the volume at
the start of the increment), so that :math:`\\sum_p w_p \\bar{v}_p = \\sum_p w_p v_p` (the projection keeps the total
volume, the basis being a partition of unity), and a homogeneous increment is reproduced exactly (patch test).

The physics is that of the cells; this module only drives the two passes and assembles the coupling, so it is
physics-agnostic. The tangent is exact: with :math:`\\boldsymbol{L}_a = \\sum_p N_a(\\boldsymbol{x}_p)\\, \\partial
\\boldsymbol{R}/\\partial \\bar{v}_p` and :math:`\\boldsymbol{G}_a = \\partial v_a/\\partial \\boldsymbol{Q} =
\\sum_p N_a(\\boldsymbol{x}_p)\\, w_p\\, \\partial v_p/\\partial \\boldsymbol{Q} / \\sum_p N_a(\\boldsymbol{x}_p)\\, w_p`,

.. math::

    \\frac{\\partial \\boldsymbol{R}}{\\partial \\boldsymbol{Q}}\\bigg|_{\\text{coupling}} = \\sum_a \\boldsymbol{L}_a
    \\otimes \\boldsymbol{G}_a ,

a rank-one block per node over the dofs of the cells that share the node (:class:`NodalProjectionCoupling`, an entity
of the discrete system without residual). The entities are constraints of the discrete system in the sense of
EdelweissFE, so they are assembled before the multi-point constraints condense the system: with the aggregated MPM, the
projection acts on the full grid, and :math:`E^T K E` reduces it.

**Aggregated MPM.** Projected independently, the values of all nodes of the full grid would impose one volumetric
constraint per node on the reduced displacement space (the exterior nodes carry no displacement dofs of their own) and
lock (cantilever of example 194: 0.025 x the Timoshenko deflection). Therefore, with an ``extension``
(:meth:`NodalProjection.updateConnectivity`, given by
:class:`~edelweissmeshfree.constraints.aggregatedmpm.AggregatedMaterialPointMethod`), the value of an exterior node is
extended from the interior nodes with the weights of the aggregation, :math:`v_s = \\sum_m E_{sm} v_m`, and the
projection uses the extended basis :math:`\\tilde{N}_m = \\sum_a N_a E_{am}` (still a partition of unity, so the
patch test and the volume conservation hold; the weights :math:`\\sum_p \\tilde N_m w_p` of the lumped projection are
positive in practice, a node without positive weight keeps :math:`v_m = 1` and no coupling). Then there is one
volumetric constraint per independent node, and the tangent keeps its symmetric, positive semidefinite volumetric part
at a zero increment.

**Choice of the projected quantity.** :math:`\\det\\Delta\\boldsymbol{F}_p` with the volumes :math:`V_p` at the
start of the increment as weights: the lumped projection then conserves the volume of the body over the increment
(:math:`\\sum_p V_p \\Delta\\bar{J}_p = \\sum_p V_p \\Delta J_p`), as the F-bar patch does, and reproduces a
homogeneous increment exactly. Averaging :math:`\\ln \\Delta J` would conserve the product, not the volume, and
differs only at second order in the increment.

**Literature.** This is the incremental F-bar with a projected (smoothed) volume change, in the spirit of the
F-bar/B-bar projection methods for particle methods (Moutsanidis et al. 2020, projection of the volumetric measure
onto the background basis) and of nodally averaged pressure/volume elements (Bonet and Burton 1998), here with the
lumped (row-sum) L2 projection onto the grid nodes and the exact linearisation.

Use, without aggregation:

.. code-block:: python

    from edelweissmeshfree.constraints.nodalprojection import NodalProjection

    mpmModel.cellProjections["nodalJ"] = NodalProjection("nodalJ", mpmModel, journal)

and with it, ``AggregatedMaterialPointMethod(..., projectionMode="nodal")``.
"""

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.timestep import TimeStep


class NodalProjectionCoupling(ConstraintBase):
    """The coupling of the cells sharing one node through the projected value of the node: an entity of the discrete
    system over the nodes of these cells, which adds :math:`\\boldsymbol{L}_a \\otimes \\boldsymbol{G}_a` to the tangent
    (no residual).

    Parameters
    ----------
    name
        The name.
    projection
        The projection the entity belongs to.
    index
        The index of the node in the projection.
    nodes
        The nodes of the cells sharing the node.
    fields
        The fields of these nodes.
    """

    #: The projection is evaluated before the cells interpolate; :attr:`projector` does it once for all entities.
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
        """The coupling through the value of the node, after the cells were computed: no residual."""

        L, G = self.projector.couplingOfNode(self._index)
        if L is not None:
            K += np.outer(L, G)


class NodalProjection:
    """The projection through the nodes over the active cells of an MPM model that support it.

    Parameters
    ----------
    name
        The name.
    model
        The :class:`~edelweissmeshfree.models.mpmmodel.MPMModel`; the active cells are those of its MPM managers.
    journal
        The journal, for a summary when the active cells change.
    """

    def __init__(self, name: str, model, journal: Journal = None):
        self._name = name
        self._model = model
        self._journal = journal
        self._activeCellNumbers = None
        self._entities = []
        self._cells = []
        self._evaluation = 0
        self._couplingEvaluation = -1
        #: The projected nodal values of the last evaluation (in the order of :attr:`nodes`).
        self.nodalValues = np.zeros(0)
        #: The nodal weights :math:`\sum_p N_a w_p` of the last evaluation.
        self.nodalWeights = np.zeros(0)
        #: The nodes of the projection, i.e. of the active cells that support it.
        self.nodes = []

    @property
    def name(self) -> str:
        return self._name

    def _message(self, message: str):
        if self._journal is not None:
            self._journal.message(message, self._name)

    def updateConnectivity(self, activeCells: list = None, extension: dict = None):
        """Follow the active cells (called by the model after its managers assigned the material points): the entities
        are built once per set of active cells, so that they are the same as long as the discrete system lives.

        Parameters
        ----------
        activeCells
            The active cells; default those of the model's MPM managers.
        extension
            Optional, for the aggregated MPM: the nodal values of the slave (exterior) nodes are not projected but
            extended from those of the master nodes, ``{slaveNodeLabel: [(masterNode, weight), ...]}`` (the scalar
            weights of :attr:`~edelweissmeshfree.numerics.aggregation.CellAggregation.extension`); it must belong to
            the same set of active cells.
        """

        if activeCells is None:
            activeCells = self._model.activeCellsOf(self._model.mpmManagers)
        activeCellNumbers = tuple(cell.cellNumber for cell in activeCells)
        if activeCellNumbers == self._activeCellNumbers:
            return
        self._activeCellNumbers = activeCellNumbers
        self._build([c for c in activeCells if getattr(c, "hasMaterialPointProjection", False)], extension or {})

    def _build(self, cells: list, extension: dict):
        dimension = self._model.domainSize
        self._cells = cells
        nodeIndex = {}
        nodeFields = {}
        for cell in cells:
            for node, fields in zip(cell.nodes, cell.fields):
                if node not in nodeIndex:
                    nodeIndex[node] = len(nodeIndex)
                    nodeFields[node] = list(fields)
                else:
                    assert nodeFields[node] == list(fields), "the cells must carry the same fields on a node"
        self.nodes = list(nodeIndex)
        nNodes = len(self.nodes)

        # slave nodes (aggregation): value_s = sum_m E_sm value_m, if all their masters take part
        indexByLabel = {n.label: i for n, i in nodeIndex.items()}
        slaves = {}
        for label, masters in extension.items():
            if label in indexByLabel and all(m in nodeIndex for m, _ in masters):
                slaves[indexByLabel[label]] = [(nodeIndex[m], float(w)) for m, w in masters]
        self._nSlaves = len(slaves)
        #: the independent nodes (indices into nodes), whose values are projected
        self._independent = [a for a in range(nNodes) if a not in slaves]
        position = {a: k for k, a in enumerate(self._independent)}

        # per cell: its independent nodes and T_c (cell node -> independent node), so that the basis of the projection
        # is N_c T_c (the extended basis, a partition of unity)
        self._cellNodeIndices = []
        self._cellMaps = []
        for cell in cells:
            columns = {}
            entries = []
            for A, node in enumerate(cell.nodes):
                a = nodeIndex[node]
                for m, w in slaves.get(a, [(a, 1.0)]):
                    k = columns.setdefault(position[m], len(columns))
                    entries.append((A, k, w))
            T = np.zeros((len(cell.nodes), len(columns)))
            for A, k, w in entries:
                T[A, k] += w
            self._cellNodeIndices.append(np.array(list(columns), dtype=int))
            self._cellMaps.append(T)
        nIndependent = len(self._independent)
        cellsOfNode = [[] for _ in range(nIndependent)]
        for c, idx in enumerate(self._cellNodeIndices):
            for k, m in enumerate(idx):
                cellsOfNode[m].append((c, k))

        # one entity per independent node m: over the nodes of the cells whose (extended) basis contains it
        self._entities = []
        self._entityParts = []
        for m in range(nIndependent):
            entityCells = sorted({c for c, _ in cellsOfNode[m]})
            entityNodes = sorted({n for c in entityCells for n in cells[c].nodes}, key=lambda n: n.label)
            offset = {}
            n = 0
            for node in entityNodes:
                offset[node] = n
                n += sum(getFieldSize(f, dimension) for f in nodeFields[node])
            parts = []
            for c, k in cellsOfNode[m]:
                cell = cells[c]
                nodeWise = []
                for node in cell.nodes:
                    nodeWise.extend(offset[node] + np.arange(sum(getFieldSize(f, dimension) for f in nodeFields[node])))
                nodeWise = np.asarray(nodeWise, dtype=int)
                perm = cell.dofIndicesPermutation
                parts.append((c, k, nodeWise[np.asarray(perm, dtype=int)] if perm is not None else nodeWise))
            self._entityParts.append(parts)
            self._entities.append(
                NodalProjectionCoupling(
                    "{:} node {:}".format(self._name, self.nodes[self._independent[m]].label),
                    self,
                    m,
                    entityNodes,
                    [nodeFields[node] for node in entityNodes],
                    n,
                )
            )
        self._message(
            "nodal projection over {:} cells, {:} nodes ({:} extended from the others)".format(
                len(cells), nNodes, self._nSlaves
            )
        )

    def couplingEntities(self) -> list:
        """The entities of the coupling, one per independent node of the current active cells that support the
        projection."""

        return list(self._entities)

    def projectToCells(self, dU):
        """Pass 1 and 2: the nodal values at the trial increment, interpolated back to every material point of every
        cell. Before the cells interpolate to their material points.

        Parameters
        ----------
        dU
            The trial solution increment of the discrete system (indexed by the cells).
        """

        nIndependent = len(self._independent)
        W = np.zeros(nIndependent)
        V = np.zeros(nIndependent)
        self._N = []
        self._G = []
        for cell, idx, T in zip(self._cells, self._cellNodeIndices, self._cellMaps):
            nP = len(cell.assignedMaterialPoints)
            nDofCell = cell.nDof
            v = np.zeros(nP)
            w = np.zeros(nP)
            dv = np.zeros(nP * nDofCell)
            N = np.zeros(nP * T.shape[0])
            cell.computeMaterialPointProjectionContributions(np.ascontiguousarray(dU[cell]), v, w, dv)
            cell.getMaterialPointProjectionBasis(N)
            N = N.reshape(nP, T.shape[0]) @ T
            Nw = N * w[:, None]
            np.add.at(W, idx, Nw.sum(axis=0))
            np.add.at(V, idx, Nw.T @ v)
            self._N.append(N)
            self._G.append(Nw.T @ dv.reshape(nP, nDofCell))  # per independent node of the cell, divided by W later
        positive = W > 0.0
        values = np.ones(nIndependent)
        values[positive] = V[positive] / W[positive]
        #: the nodal values and weights of the last evaluation, on the independent nodes
        self.nodalValues = values
        self.nodalWeights = W
        for cell, idx, N in zip(self._cells, self._cellNodeIndices, self._N):
            cell.setMaterialPointProjectedValues(np.ascontiguousarray(N @ values[idx]))
        self._evaluation += 1

    def _updateSensitivities(self):
        """The derivatives of the residuals of the cells w.r.t. the nodal values, :math:`\\sum_p N_m \\partial
        \\boldsymbol{R}_c/\\partial \\bar{v}_p`, per cell: computed once per evaluation, after the cells."""

        self._L = []
        for cell, N in zip(self._cells, self._N):
            nP = N.shape[0]
            S = np.zeros(nP * cell.nDof)
            cell.getMaterialPointProjectionSensitivities(S)
            self._L.append(N.T @ S.reshape(nP, cell.nDof))
        self._couplingEvaluation = self._evaluation

    def couplingOfNode(self, m: int) -> tuple:
        """The vectors :math:`\\boldsymbol{L}_m, \\boldsymbol{G}_m` of independent node ``m`` over the dofs of its
        entity, or ``(None, None)`` for a node without positive weight."""

        if self._couplingEvaluation != self._evaluation:
            self._updateSensitivities()
        W = self.nodalWeights[m]
        if W <= 0.0:
            return None, None
        nDof = self._entities[m].nDof
        L = np.zeros(nDof)
        G = np.zeros(nDof)
        for c, k, dofs in self._entityParts[m]:
            np.add.at(L, dofs, self._L[c][k])
            np.add.at(G, dofs, self._G[c][k])
        return L, G / W
