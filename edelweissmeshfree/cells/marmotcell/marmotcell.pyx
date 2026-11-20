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

import numpy as np

cimport cython
cimport libcpp.cast
cimport numpy as np

cimport edelweissmeshfree.cells.marmotcell.marmotcell

from edelweissfe.numerics.systemmatrixcontributor import denseBlockLocalEntries
from edelweissfe.utils.exceptions import CutbackRequest

from libc.stdlib cimport free, malloc
from libcpp.memory cimport allocator, make_unique, unique_ptr

from edelweissmeshfree.materialpoints.marmotmaterialpoint.mp cimport (
    MarmotMaterialPointWrapper,
)


@cython.final # no subclassing -> cpdef with nogil possible
cdef class MarmotCellWrapper:
    """This cell as a wrapper for MarmotCells.
    It is an abstract class and cannot be used directly.
    Rather, it is used as a base class for the specific cell types, such as LagrangianMarmotCell and BSplineMarmotCell.

    For the documentation of MarmotCells, please refer to `Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot/>`_.

    Parameters
    ----------
    cellType
        The Marmot element which should be represented, e.g., CPE4.
    cellNumber
        The (unique) number of this cell.
    nodes
        The list of nodes for this Cell.
        """


    def __init__(self, cellType, cellNumber, nodes):
        self._cellNumber = cellNumber
        self._cellType = cellType

        self._nNodes                         = self._marmotCell.getNNodes()

        self._nDof                           = self._marmotCell.getNDofPerCell()

        cdef vector[vector[string]] fields  = self._marmotCell.getNodeFields()
        self._fields                         = [ [ s.decode('utf-8')  for s in n  ] for n in fields ]

        cdef vector[int] permutationPattern = self._marmotCell.getDofIndicesPermutationPattern()
        self._dofIndicesPermutation          = np.asarray(permutationPattern)

        cdef dict supportedBodyLoads = self._marmotCell.getSupportedBodyLoadTypes()
        self._supportedBodyLoads = {k.decode() :  v for k, v in supportedBodyLoads.items()  }

        cdef dict supportedDistributedLoads = self._marmotCell.getSupportedDistributedLoadTypes()
        self._supportedDistributedLoads = {k.decode() :  v for k, v in supportedDistributedLoads.items()  }

        self._ensightType                    = self._marmotCell.getCellShape().decode('utf-8')

    @property
    def cellNumber(self):
        return self._cellNumber

    @property
    def cellType(self):
        return self._cellType

    @property
    def nodes(self):
        return self._nodes

    @property
    def nNodes(self):
        return self._nNodes

    @property
    def nDof(self):
        return self._nDof

    @property
    def fields(self):
        return self._fields

    @property
    def dofIndicesPermutation(self):
        return self._dofIndicesPermutation

    @property
    def ensightType(self):
        return self._ensightType

    @property
    def assignedMaterialPoints(self):
        return self._assignedMaterialPoints

    def computeKernels(self,
                       double[::1] Kc,
                       double[::1] Pc,
                       const double[::1] Uc,
                       const double[::1] dUc,
                       double timeNew,
                       double dTime):
        """Add the internal forces of the assigned material points to the residual, and their tangent; see
        :meth:`edelweissmeshfree.cells.base.cell.CellBase.computeKernels`. A cell is incremental, so Marmot gets the
        solution increment. A failure in Marmot requests a cutback, as for an EdelweissFE element."""

        try:
            with nogil:
                self._marmotCell.computeMaterialPointKernels(&dUc[0], &Pc[0], &Kc[0], timeNew, dTime)
        except (RuntimeError, ValueError) as e:
            raise CutbackRequest(str(e), 0.5)

    cpdef void interpolateFieldsToMaterialPoints(self, double[::1] dUc) nogil:

        self._marmotCell.interpolateFieldsToMaterialPoints(&dUc[0])

    cpdef void computeLumpedInertia( self, double[::1] M ) nogil:
        self._marmotCell.computeLumpedInertia(&M[0])


    cpdef void computeConsistentInertia( self, double[::1] M ) nogil:
        self._marmotCell.computeConsistentInertia(&M[0])

    @property
    def connectivityChangesWithMotion(self):
        """False: the nodes of a cell stay."""
        return False

    @property
    def scalarVariables(self):
        """No scalar variables, as for every cell; see
        :attr:`edelweissfe.nodecouplingentity.base.nodecouplingentity.BaseNodeCouplingEntity.scalarVariables`."""
        return ()

    def computeBodyLoad(self,
                         str loadType,
                         double[::1] load,
                         double[::1] Pc,
                         double[::1] Kc,
                         double timeNew,
                         double dTime):

        self._marmotCell.computeBodyLoad( self._supportedBodyLoads[loadType.upper()], &load[0], &Pc[0], &Kc[0], timeNew, dTime)

    def computeDistributedLoad(self,
                         str loadType,
                         int surfaceID,
                         materialPoint,
                         double[::1] load,
                         double[::1] Pc,
                         double[::1] Kc,
                         double timeNew,
                         double dTime):

        cdef int mpNumber = materialPoint.number

        self._marmotCell.computeDistributedLoad( self._supportedDistributedLoads[loadType.upper()],
                                                surfaceID,
                                                mpNumber,
                                                &load[0],
                                                &Pc[0],
                                                &Kc[0],
                                                timeNew,
                                                dTime)

    def assignMaterialPoints(self, list marmotMaterialPointWrappers):
        cdef vector[MarmotMaterialPoint*] mps
        cdef MarmotMaterialPointWrapper mpWrapper

        for mpWrapper in marmotMaterialPointWrappers:
            mps.push_back(<MarmotMaterialPoint*> mpWrapper._marmotMaterialPoint)

        self._marmotCell.assignMaterialPoints(mps)

        self._assignedMaterialPoints = marmotMaterialPointWrappers

    def setProjectionPoint(self, coordinate: np.ndarray) -> bool:
        """Set the point (physical coordinates) at which a cell-level projection of the cell (e.g. the volume change
        of an F-bar cell) is evaluated, until the next :meth:`assignMaterialPoints`; see the Marmot cell. Used by the
        aggregated material point method to evaluate the projection of all cells of an aggregate at one point.

        Returns
        -------
        bool
            Whether the cell uses the point; ``False`` for cells without such a projection.
        """
        cdef double[::1] coords = np.ascontiguousarray(coordinate, dtype=float)
        return self._marmotCell.setProjectionPoint(&coords[0])

    @property
    def hasPatchProjection(self) -> bool:
        """Whether the cell supports a projection over a patch of cells (e.g. F-bar patch), see
        :class:`~edelweissmeshfree.constraints.cellpatchprojection.CellPatchProjection`."""
        return self._marmotCell.hasPatchProjection()

    def computeProjectionContribution(self, double[::1] dUc, double[::1] dValue_dQ) -> tuple[float, float]:
        """Pass 1 of the patch projection: the contribution of the material points of the cell at the increment
        ``dUc`` (cell dof order). Writes its derivative to ``dValue_dQ``; changes no material point.

        Returns
        -------
        tuple[float, float]
            The weighted value and the weight.
        """
        cdef double value = 0.0, weight = 0.0
        self._marmotCell.computeProjectionContribution(&dUc[0], &value, &weight, &dValue_dQ[0])
        return value, weight

    def setProjectedValue(self, double value):
        """Pass 2 of the patch projection: the value of the patch the next evaluations use."""
        self._marmotCell.setProjectedValue(value)

    def getProjectionSensitivity(self, double[::1] dR_dValue):
        """The derivative of the residual contributions of the cell w.r.t. the patch value (cell dof order)."""
        self._marmotCell.getProjectionSensitivity(&dR_dValue[0])

    @property
    def hasMaterialPointProjection(self) -> bool:
        """Whether the cell supports a projection with one value per material point (e.g. nodal F-bar), see
        :class:`~edelweissmeshfree.constraints.nodalprojection.NodalProjection`."""
        return self._marmotCell.hasMaterialPointProjection()

    def computeMaterialPointProjectionContributions(self, double[::1] dUc, double[::1] values, double[::1] weights,
                                                    double[::1] dValues_dQ):
        """Pass 1 of the projection per material point: the values, weights and derivatives (material point major, cell
        dof order) of the assigned material points at the increment ``dUc``; changes no material point. Does nothing
        without material points."""
        if values.shape[0] == 0:
            return
        self._marmotCell.computeMaterialPointProjectionContributions(&dUc[0], &values[0], &weights[0], &dValues_dQ[0])

    def getMaterialPointProjectionBasis(self, double[::1] N):
        """The cell basis at the assigned material points (material point major), as the cell uses it."""
        if N.shape[0] == 0:
            return
        self._marmotCell.getMaterialPointProjectionBasis(&N[0])

    def setMaterialPointProjectedValues(self, double[::1] values):
        """Pass 2 of the projection per material point: the values the next evaluations use."""
        if values.shape[0] == 0:
            return
        self._marmotCell.setMaterialPointProjectedValues(&values[0])

    def getMaterialPointProjectionSensitivities(self, double[::1] dR_dValues):
        """The derivatives of the residual contributions of the cell w.r.t. the projected values (material point
        major, cell dof order)."""
        if dR_dValues.shape[0] == 0:
            return
        self._marmotCell.getMaterialPointProjectionSensitivities(&dR_dValues[0])

    def isCoordinateInCell(self, coordinate: np.ndarray) -> bool:
        cdef double[::1] coords = coordinate
        return self._marmotCell.isCoordinateInCell(&coords[0])

    def getBoundingBox(self, ):

        cdef int dim = self._nodeCoordinates.shape[1]
        cdef np.ndarray boundingBoxMin = np.zeros(dim)
        cdef np.ndarray boundingBoxMax = np.zeros(dim)
        cdef double[::1] boundingBoxMinView = boundingBoxMin
        cdef double[::1] boundingBoxMaxView = boundingBoxMax

        self._marmotCell.getBoundingBox(&boundingBoxMinView[0], &boundingBoxMaxView[0])

        return boundingBoxMin, boundingBoxMax

    def getInterpolationVector(self, coordinate: np.ndarray) -> np.ndarray:
        cdef double[::1] coords = coordinate
        cdef np.ndarray N = np.zeros(self._nNodes)
        cdef double[::1] Nview_ = N
        self._marmotCell.getInterpolationVector(&Nview_[0], &coords[0])
        return N

    def getInterpolationGradient(self, coordinate: np.ndarray) -> np.ndarray:
        """The gradients of the shape functions w.r.t. the physical coordinates at a point, also outside of the cell
        (the gradients of the polynomial extension of :meth:`getInterpolationVector`).

        Parameters
        ----------
        coordinate
            The coordinates of the point.

        Returns
        -------
        np.ndarray
            :math:`\\partial N_a / \\partial x_i`, shape ``(nDim, nNodes)``.

        Raises
        ------
        RuntimeError
            If the cell does not provide its interpolation gradients, or the inverse map fails.
        """
        cdef double[::1] coords = coordinate
        cdef int dim = self._nodeCoordinates.shape[1]
        cdef np.ndarray dNdX = np.zeros(self._nNodes * dim)
        cdef double[::1] dNdXview_ = dNdX
        self._marmotCell.getInterpolationGradient(&dNdXview_[0], &coords[0])
        return dNdX.reshape((self._nNodes, dim)).T.copy()

    @property
    def couplesAllItsDofs(self) -> bool:
        """``True``: the block is the dense ``nDof x nDof`` matrix, column-major."""
        return True

    def localEntriesOfBlock(self) -> tuple[np.ndarray, np.ndarray]:
        """The entries of the dense block, see
        :func:`~edelweissfe.numerics.systemmatrixcontributor.denseBlockLocalEntries`."""
        return denseBlockLocalEntries(self.nDof)

    def shapeBlock(self, values: np.ndarray) -> np.ndarray:
        """Keep the view flat for Cython 1-D memoryview compatibility."""
        return values

