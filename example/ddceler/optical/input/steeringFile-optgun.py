# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""DDG4 steering file for optical photon propagation validation.

Fires monochromatic optical photons directly into the scintillating water
box so that Celeritas (direct-offload mode) and pure Geant4 can be compared
on identical input — isolating propagation from generation differences.
"""

from DDSim.DD4hepSimulation import DD4hepSimulation

runner = DD4hepSimulation()

runner.action.run = "CelerRun"
runner.action.tracker = "Geant4TrackerAction"
runner.action.trackerSDTypes = ["tracker"]
runner.action.calo = "Geant4CalorimeterAction"
runner.action.calorimeterSDTypes = ["calorimeter"]

runner.outputConfig.forceDD4HEP = True
runner.numberOfEvents = 100

# Field tracking configuration
runner.field.delta_chord = 0.025  # mm
runner.field.delta_intersection = 1e-2  # mm
runner.field.delta_one_step = 0.001  # mm
runner.field.eps_min = 5e-5  # mm
runner.field.eps_max = 0.001  # mm
runner.field.min_chord_step = 1e-6  # mm

# 3.1 eV (400 nm) optical photon gun, 1 mm inside the water box front face
runner.enableGun = True
runner.gun.particle = "opticalphoton"
runner.gun.energy = 3.1e-3  # MeV (= 3.1 eV)
runner.gun.direction = (0, 0, 1)
runner.gun.position = (0, 0, -149)  # mm
runner.gun.multiplicity = 1000


def setup_physics(kernel):
    """Configure optical physics list with Celeritas integration."""
    from DDG4 import Geant4, PhysicsList

    phys = Geant4(kernel).setupPhysics("QGSP_BERT")

    # Optical photon processes: boundary, absorption, Rayleigh
    ph = PhysicsList(kernel, "Geant4OpticalPhotonPhysics/OpticalGammaPhys")
    ph.VerboseLevel = 1
    ph.addParticleConstructor("G4OpticalPhoton")
    ph.enableUI()
    phys.adopt(ph)

    # No Cherenkov or scintillation — photons are injected directly

    # Celeritas offload with optical tracking (direct mode)
    celer_phys = PhysicsList(kernel, str("CelerPhysics"))
    celer_phys.MaxNumTracks = 2048
    celer_phys.InitCapacity = 245760
    celer_phys.OpticalTracks = 2048
    celer_phys.OpticalGenerator = "direct"
    phys.adopt(celer_phys)
    phys.dump()
    return None


runner.physics.setupUserPhysics(setup_physics)
