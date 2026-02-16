import numpy as np
from edelweissfe.journal.journal import Journal
from edelweissmeshfree.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import KDBinOrganizedParticleManager

class RemoveDamagedParticleManager(KDBinOrganizedParticleManager):
    """
    A particle manager that removes particles and their associated kernels 
    when damage exceeds a specified threshold.
    """
    def __init__(
        self,
        particleKernelDomain: ParticleKernelDomain,
        dimension: int,
        journal: Journal,
        damageThreshold: float = 0.9,
        **kwargs
    ):
        super().__init__(particleKernelDomain, dimension, journal, **kwargs)
        self._damageThreshold = damageThreshold
        self._particleKernelDomain = particleKernelDomain

    def updateConnectivity(self) -> bool:
        active_particles = []
        removed_kernels = set()
        any_removed = False

        for p in self._particles:
            try:
                damage_val = p.getResultArray("omega", getPersistentView=False)
                if np.any(damage_val >= self._damageThreshold):
                    self._journal.message(f"Removing particle {p.number}", "DamageManager")
                    removed_kernels.update(p.kernelFunctions)
                    any_removed = True
                    continue
            except (KeyError, AttributeError):
                pass
            active_particles.append(p)

        if any_removed:
            # Update Domain in-place
            self._particleKernelDomain.particles.clear()
            self._particleKernelDomain.particles.extend(active_particles)
            self._particles = active_particles

            # Update Kernels in-place
            remaining_kernels = [k for k in self._particleKernelDomain.meshfreeKernelFunctions
                                if k not in removed_kernels]
            self._particleKernelDomain.meshfreeKernelFunctions.clear()
            self._particleKernelDomain.meshfreeKernelFunctions.extend(remaining_kernels)
            self._meshfreeKernelFunctions = remaining_kernels

            self.signalizeKernelFunctionUpdate()

        # The 'True' return here triggers the Solver to rebuild the DofManager
        return any_removed or super().updateConnectivity()
        #"""
        #Filters out highly damaged particles and their unique kernels.
        #"""
        #active_particles = []
        #removed_kernels = set()
        #any_removed = False

        #for p in self._particles:
        #    try:
        #        # Damage is typically retrieved from the material point's state variables
        #        damage_val = p.getResultArray("omega", getPersistentView=False)
        #        
        #        if np.any(damage_val >= self._damageThreshold):
        #            self._journal.message(f"Removing particle {p.number} (Damage: {damage_val})", "DamageManager")
        #            removed_kernels.update(p.kernelFunctions)
        #            any_removed = True
        #            continue
        #    except (KeyError, AttributeError):
        #        # Material does not support damage; keep the particle
        #        pass
        #    
        #    active_particles.append(p)

        #if any_removed:
        #    # Update the Domain lists in-place to avoid AttributeError on read-only properties
        #    self._particleKernelDomain.particles.clear()
        #    self._particleKernelDomain.particles.extend(active_particles)
        #    self._particles = active_particles
        #    
        #    # Filter kernels to remove those belonging to dead particles
        #    remaining_kernels = [
        #        k for k in self._particleKernelDomain.meshfreeKernelFunctions 
        #        if k not in removed_kernels
        #    ]
        #    self._particleKernelDomain.meshfreeKernelFunctions.clear()
        #    self._particleKernelDomain.meshfreeKernelFunctions.extend(remaining_kernels)
        #    self._meshfreeKernelFunctions = remaining_kernels
        #    
        #    # Re-initialize the spatial binning for remaining kernels
        #    self.signalizeKernelFunctionUpdate()

        ## Perform standard spatial connectivity update
        #spatialChange = super().updateConnectivity()
        #return any_removed or spatialChange





        #"""
        #Filters out highly damaged particles before performing the 
        #standard connectivity update.
        #"""
        #initial_count = len(self._particles)
        #active_particles = []
        #removed_kernels = set()

        #for p in self._particles:
        #    # Check if particle has damage via its result/state interface
        #    # We assume the result name 'damage' is used in the material point's state manager
        #    try:
        #        # Retrieve damage from the particle's underlying material response
        #        damage_val = p.getResultArray("omega", getPersistentView=False)
        #        
        #        # Check threshold and if the material supports damage 
        #        # (implied if 'damage' array exists and is > 0)
        #        if np.any(damage_val >= self._damageThreshold):
        #            self._journal.message(f"Removing particle {p.number} (Damage: {damage_val})", "DamageManager")
        #            # Track associated kernel functions to remove them from the domain
        #            removed_kernels.update(p.kernelFunctions)
        #            continue
        #    except (KeyError, AttributeError):
        #        # Material likely doesn't allow damage; keep the particle
        #        pass
        #    
        #    active_particles.append(p)

        #if len(active_particles) < initial_count:
        #    # Update the lists in the domain to stop tracking these objects
        #    self._particles = active_particles
        #    self._particleKernelDomain.particles = active_particles
        #    
        #    # Filter kernels: keep only those not associated with deleted particles
        #    # Note: In some formulations, kernels might be shared; check if other active particles use them
        #    remaining_kernels = [
        #        k for k in self._particleKernelDomain.meshfreeKernelFunctions 
        #        if k not in removed_kernels
        #    ]
        #    self._particleKernelDomain.meshfreeKernelFunctions = remaining_kernels
        #    self._meshfreeKernelFunctions = remaining_kernels
        #    
        #    # Re-initialize binning for the reduced kernel set
        #    self.signalizeKernelFunctionUpdate()
        #    hasChanged = True
        #else:
        #    hasChanged = False

        ## Run the standard spatial connectivity update for remaining particles
        #spatialChange = super().updateConnectivity()
        #return hasChanged or spatialChange
