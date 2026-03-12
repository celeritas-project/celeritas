# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""DDG4 steering file for the optical photon demonstration.

A 1 GeV e- enters a 30 cm scintillating water cube, initiating an EM
shower (X0 ~ 36 cm in water).  All shower secondaries are above the
Cherenkov threshold (~0.26 MeV kinetic energy in water), producing a
dense cone of optical photons.  Scintillation photons are generated
along the shower axis proportional to the local energy deposit.
All optical photons are offloaded to Celeritas for GPU tracking.
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

# 1 GeV e-: EM shower mostly contained within the 30 cm water box
runner.enableGun = True
runner.gun.particle = "e-"
runner.gun.energy = "1*GeV"
runner.gun.distribution = "uniform"
runner.gun.etaMin = 5.0
runner.gun.etaMax = 5.0
runner.gun.position = "0 0 -155*mm"


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

    # Cherenkov radiation from shower tracks.
    # MaxNumPhotonsPerStep is kept low to avoid exhausting the optical
    # generator buffer given the high shower track multiplicity.
    ph = PhysicsList(kernel, "Geant4CerenkovPhysics/CerenkovPhys")
    ph.MaxNumPhotonsPerStep = 50
    ph.MaxBetaChangePerStep = 10.0
    ph.TrackSecondariesFirst = False
    ph.VerboseLevel = 1
    ph.enableUI()
    phys.adopt(ph)

    # Scintillation along the shower axis (yield = 100/MeV in geometry XML)
    ph = PhysicsList(kernel, "Geant4ScintillationPhysics/ScintillatorPhys")
    ph.ScintillationYieldFactor = 1.0
    ph.ScintillationExcitationRatio = 1.0
    ph.TrackSecondariesFirst = False
    ph.VerboseLevel = 1
    ph.enableUI()
    phys.adopt(ph)

    # Celeritas offload with optical tracking
    celer_phys = PhysicsList(kernel, str("CelerPhysics"))
    celer_phys.MaxNumTracks = 2048
    celer_phys.InitCapacity = 245760
    celer_phys.IgnoreProcesses = ["CoulombScat"]
    # OpticalGenerators defaults to OpticalTracks * 8 = 16384
    celer_phys.OpticalTracks = 2048
    phys.adopt(celer_phys)
    phys.dump()
    return None


runner.physics.setupUserPhysics(setup_physics)
runner.part.userParticleHandler = ""
