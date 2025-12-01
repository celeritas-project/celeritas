# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Minimal DDG4 steering file for Geant4 with Celeritas integration.

Usage:
    ddsim --compactFile=$PWD/example/ddceler/SiD_ConstantField.xml \\
          --steering $PWD/example/ddceler/steeringFile.py \\
          --outputFile output.root
"""

from DDSim.DD4hepSimulation import DD4hepSimulation

runner = DD4hepSimulation()

# Action configuration
runner.action.run = "DDcelerRunAction"
runner.action.tracker = "Geant4TrackerAction"
runner.action.trackerSDTypes = ["tracker"]
runner.action.calo = "Geant4CalorimeterAction"
runner.action.calorimeterSDTypes = ["calorimeter"]

# Output configuration
runner.outputConfig.forceDD4HEP = True

# Number of events
runner.numberOfEvents = 20

# Particle gun configuration
runner.enableGun = True
runner.gun.particle = "e-"
runner.gun.energy = "5*GeV"
runner.gun.distribution = "uniform"
runner.gun.etaMin = 1
runner.gun.etaMax = 2

# Field tracking configuration - defined once, used by both
# DD4hep/Geant4 and Celeritas
runner.field.delta_chord = 0.025  # mm
runner.field.delta_intersection = 1e-5  # mm
runner.field.delta_one_step = 0.01  # mm
runner.field.eps_min = 5e-5  # mm
runner.field.eps_max = 0.001  # mm
runner.field.min_chord_step = 1e-6  # mm


def setup_physics(kernel):
    """Configure physics list with Celeritas integration."""
    from DDG4 import Geant4, PhysicsList

    phys = Geant4(kernel).setupPhysics("QGSP_BERT")
    celer_phys = PhysicsList(kernel, str("DDcelerTMI"))
    celer_phys.MaxNumTracks = 2048
    celer_phys.InitCapacity = 245760
    # Celeritas does not support EmStandard MSC physics above 200 MeV
    celer_phys.IgnoreProcesses = ["CoulombScat"]
    phys.adopt(celer_phys)
    phys.dump()
    return None


runner.physics.setupUserPhysics(setup_physics)

runner.part.userParticleHandler = ""
