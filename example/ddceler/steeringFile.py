"""

Minimal proof of concept for DDG4 steering file to run Geant4 with celeritas integration.

Usage:

ddsim --compactFile=$PWD/example/ddceler/SiD_ConstantField.xml --steering $PWD/example/ddceler/steeringFile.py --outputFile output.root

"""

from DDSim.DD4hepSimulation import DD4hepSimulation

RUNNER = DD4hepSimulation()

# Action configuration
RUNNER.action.run = "DDcelerRunAction"
RUNNER.action.tracker = "Geant4TrackerAction"
RUNNER.action.trackerSDTypes = ["tracker"]
RUNNER.action.calo = "Geant4CalorimeterAction"
RUNNER.action.calorimeterSDTypes = ["calorimeter"]

# Output configuration
RUNNER.outputConfig.forceDD4HEP = True

# Number of events
RUNNER.numberOfEvents = 20

# Particle gun configuration
RUNNER.enableGun = True
RUNNER.gun.particle = "e-"
RUNNER.gun.energy = "5*GeV"
RUNNER.gun.distribution = "uniform"
RUNNER.gun.etaMin = 1
RUNNER.gun.etaMax = 2

# Field tracking configuration - defined once, used by both DD4hep/Geant4 and Celeritas
RUNNER.field.delta_chord = 0.025  # mm
RUNNER.field.delta_intersection = 1e-5  # mm
RUNNER.field.delta_one_step = 0.01  # mm
RUNNER.field.eps_min = 5e-5  # mm
RUNNER.field.eps_max = 0.001  # mm
RUNNER.field.min_chord_step = 1e-6  # mm

# Physics configuration
def setupPhysics(kernel):
  from DDG4 import PhysicsList,Geant4
  phys = Geant4(kernel).setupPhysics('QGSP_BERT')
  celer_phys = PhysicsList(kernel, str('DDcelerTMI'))
  celer_phys.MaxNumTracks = 2048
  celer_phys.InitCapacity = 245760
  # Celeritas does not support EmStandard MSC physics above 200 MeV
  celer_phys.IgnoreProcesses = ["CoulombScat"]
  phys.adopt(celer_phys)
  phys.dump()
  return None

RUNNER.physics.setupUserPhysics(setupPhysics)

RUNNER.part.userParticleHandler=''
