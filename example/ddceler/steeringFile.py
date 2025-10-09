"""
Minimal proof of concept for DDG4 steering file to run Geant4 with celeritas integration.
Usage:
CELER_LOG=debug CELER_LOG_LOCAL=debug ddsim --action.run DDcelerRunAction --compactFile=$PWD/install/share/CeleritasDD4hep/celeritas-dd4hep.xml --steering steeringFile.py -N 500 -G --gun.particle e- --gun.etaMax 4 --gun.etaMin 1 --gun.energy 5*GeV --gun.distribution uniform --outputFile output.edm4hep.root --outputConfig.forceDD4HEP  --action.tracker "Geant4TrackerAction" --action.trackerSDTypes "tracker" --action.calorimeter "Geant4CalorimeterAction" --action.calorimeterSDTypes "calorimeter"
"""
from DDSim.DD4hepSimulation import DD4hepSimulation
RUNNER = DD4hepSimulation()

# Parameterize the uniform field strength (Tesla)
UNIFORM_FIELD_STRENGTH = 4.0

# Physics configuration
def setupPhysics(kernel):
  from DDG4 import PhysicsList, Geant4
  import dd4hep

  # Override the field first, before setting up physics
  description = kernel.detectorDescription()

  # Create a constant uniform field in Z direction
  field = dd4hep.ConstantField(description, "UniformField",
                                dd4hep.Direction(0, 0, UNIFORM_FIELD_STRENGTH))

  # Set this as the new field
  description.setField(field)

  # Now setup physics
  phys = Geant4(kernel).setupPhysics('QGSP_BERT')
  celer_phys = PhysicsList(kernel, str('DDcelerTMI'))
  celer_phys.MaxNumTracks = 2048
  celer_phys.InitCapacity = 245760
  celer_phys.UniformFieldStrength = UNIFORM_FIELD_STRENGTH
  phys.adopt(celer_phys)
  phys.dump()
  return None

RUNNER.physics.setupUserPhysics(setupPhysics)
RUNNER.part.userParticleHandler=''
