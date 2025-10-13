"""

Minimal proof of concept for DDG4 steering file to run Geant4 with celeritas integration.

Usage:

CELER_LOG=debug CELER_LOG_LOCAL=debug ddsim --action.run DDcelerRunAction --compactFile=$PWD/install/share/CeleritasDD4hep/celeritas-dd4hep.xml --steering steeringFile.py -N 500 -G --gun.particle e- --gun.etaMax 4 --gun.etaMin 1 --gun.energy 5*GeV --gun.distribution uniform --outputFile output.edm4hep.root --outputConfig.forceDD4HEP  --action.tracker "Geant4TrackerAction" --action.trackerSDTypes "tracker" --action.calorimeter "Geant4CalorimeterAction" --action.calorimeterSDTypes "calorimeter"

"""

from DDSim.DD4hepSimulation import DD4hepSimulation

RUNNER = DD4hepSimulation()

# Physics configuration
def setupPhysics(kernel):
  from DDG4 import PhysicsList,Geant4
  import dd4hep
  field = kernel.detectorDescription().field()
  uniform_field_strength = field.magneticField((0,0,0))
  print("Warning: Celeritas will assume a constant field everywhere that matches the field value at origin as specified in dd4hep xml description.")
  phys = Geant4(kernel).setupPhysics('QGSP_BERT')
  celer_phys = PhysicsList(kernel, str('DDcelerTMI'))
  celer_phys.MaxNumTracks = 2048
  celer_phys.InitCapacity = 245760
  celer_phys.UniformFieldStrength = [uniform_field_strength.X()/dd4hep.tesla,
                                      uniform_field_strength.Y()/dd4hep.tesla,
                                      uniform_field_strength.Z()/dd4hep.tesla]
  phys.adopt(celer_phys)
  phys.dump()
  return None

RUNNER.physics.setupUserPhysics(setupPhysics)

RUNNER.part.userParticleHandler=''
