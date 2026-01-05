# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""DDG4 steering file for Preshower benchmark with Celeritas integration."""

from DDSim.DD4hepSimulation import DD4hepSimulation

runner = DD4hepSimulation()

# Action configuration
runner.action.run = "CelerRun"
runner.action.tracker = "Geant4TrackerAction"
runner.action.trackerSDTypes = ["tracker"]
runner.action.calo = "Geant4CalorimeterAction"
runner.action.calorimeterSDTypes = ["calorimeter"]

# Output configuration
runner.outputConfig.forceDD4HEP = True

# Number of events
runner.numberOfEvents = 100

# Particle gun configuration
runner.enableGun = True
runner.gun.particle = "e-"
runner.gun.energy = "5*GeV"
runner.gun.distribution = "uniform"
runner.gun.etaMin = 5.0
runner.gun.etaMax = 5.0

# Field tracking configuration - defined once, used by both
# DD4hep/Geant4 and Celeritas
runner.field.delta_chord = 0.025  # mm
runner.field.delta_intersection = 1e-2  # mm
runner.field.delta_one_step = 0.001  # mm
runner.field.eps_min = 5e-5  # mm
runner.field.eps_max = 0.001  # mm
runner.field.min_chord_step = 1e-6  # mm


def setup_physics(kernel):
    """Configure physics list with Celeritas integration."""
    from DDG4 import Geant4, PhysicsList

    phys = Geant4(kernel).setupPhysics("QGSP_BERT")
    celer_phys = PhysicsList(kernel, str("CelerPhysics"))
    # MaxNumTracks: max number of tracks in flight
    # InitCapacity: initial capacity for state data allocation
    # CPU defaults: MaxNumTracks=2048, InitCapacity=245760
    # GPU recommendation: MaxNumTracks=262144, InitCapacity=8388608
    celer_phys.MaxNumTracks = 2048
    celer_phys.InitCapacity = 245760
    # Celeritas does not support single scattering
    celer_phys.IgnoreProcesses = ["CoulombScat"]
    phys.adopt(celer_phys)
    phys.dump()
    return None


runner.physics.setupUserPhysics(setup_physics)

runner.part.userParticleHandler = ""
