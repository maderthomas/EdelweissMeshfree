import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmpm.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmpm.materialpoints.base.mp import MaterialPointBase
from edelweissmpm.models.mpmmodel import MPMModel

class LagrangianWeakDirichlet(MPMConstraintBase):
    """
    This is an implementation of weak Dirichlet boundary conditions using a penalty formulation.
    It constrains a material point field increment.

    Parameters
    ----------
    name
        The name of this constraint.
    constrainedMaterialPoint
        The material point field value to be constrained.
    field
        The field this constraint is acting on.
    prescribedStepDelta
        A dictionary mapping field component indices to their prescribed step increments.
    model
        The full MPMModel instance.
    """

    def __init__(
        self,
        name: str,
        constrainedMaterialPoint: MaterialPointBase,
        field: str,
        prescribedStepDelta: dict,
        model: MPMModel,
    ):
        self._name = name
        self._constrainedMP = constrainedMaterialPoint
        self._field = field
        self._prescribedStepDelta = prescribedStepDelta
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._nodes = dict()

        self._nLagrangianMultipliers = len(self._prescribedStepDelta)
        self.reactionForce = np.zeros(self._fieldSize)
        #self._lagrangianMultipliers = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list:
        return self._nodes.keys()

    @property
    def fieldsOnNodes(self) -> list:
        return [
            [
                self._field,
            ]
        ] * len(self._nodes)

    @property
    def nDof(self) -> int:
        return len(self._nodes) * self._fieldSize + self._nLagrangianMultipliers

    
    @property
    def scalarVariables(
        self,
    ) -> list:
        return self._lagrangianMultipliers
    #return []

    def getNumberOfAdditionalNeededScalarVariables(
        self,
    ) -> int:
    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        self._lagrangianMultipliers = scalarVariables
        #pass
    
    def updateConnectivity(self, model):

        nodes = {n: i for i, n in enumerate(set(n for c in self._constrainedMP.assignedCells for n in c.nodes))}

        hasChanged = False
        if nodes != self._nodes:
            hasChanged = True

        self._nodes = nodes

        return hasChanged

    def applyConstraint(self, dU_: np.ndarray, PExt: np.ndarray, V: np.ndarray, timeStep: TimeStep):

        dU_U = dU_[: -self._nLagrangianMultipliers]
        dU_L = dU_[-self._nLagrangianMultipliers :]
        PExt_U = PExt[: -self._nLagrangianMultipliers]
        PExt_L = PExt[-self._nLagrangianMultipliers :]

        K = V.reshape((self.nDof, self.nDof))

        # K_UU = K[:-self._nLagrangianMultipliers, :-self._nLagrangianMultipliers]
        K_UL = K[: -self._nLagrangianMultipliers, -self._nLagrangianMultipliers :]
        K_LU = K[-self._nLagrangianMultipliers :, : -self._nLagrangianMultipliers]
        # K_LL = K[-self._nLagrangianMultipliers:, -self._nLagrangianMultipliers:]

        self.reactionForce.fill(0.0)
        mp = self._constrainedMP

        constrainedCoordinates = mp.getCenterCoordinates()

        for i, prescribedComponent in self._prescribedStepDelta.items():

            P_U_i = PExt_U[i :: self._fieldSize]
            dU_U_j = dU_U[i :: self._fieldSize]

            K_UL_j = K_UL[i :: self._fieldSize, :]
            K_LU_j = K_LU[:, i :: self._fieldSize]

            dL_i = dU_L[i]

            for c in mp.assignedCells:
                N = c.getInterpolationVector(constrainedCoordinates)
                nodeIdcs = [self._nodes[n] for n in c.nodes]

                mpValue = N @ dU_U_j[nodeIdcs]

                g_i = mpValue - prescribedComponent * timeStep.stepProgressIncrement
                dg_i_dU_j = N

                P_U_i[nodeIdcs] += dL_i * dg_i_dU_j
                PExt_L[i] += g_i

                K_UL_j[nodeIdcs, i] += dg_i_dU_j
                K_LU_j[i, nodeIdcs] += dg_i_dU_j

            self.reactionForce[i] += dL_i


def LagrangianWeakDirichletOnMaterialPointSetFactory(
    baseName: str,
    materialPointSet: list[MaterialPointBase],
    field: str,
    prescribedStepDelta: dict,
    model: MPMModel,
):
    constraints = dict()
    for i, mp in enumerate(materialPointSet):
        name = f"{baseName}_{i}"
        constraint = LagrangianWeakDirichlet(name, p, field, prescribedStepDelta, model)
        constraints[name] = constraint

    return constraints
