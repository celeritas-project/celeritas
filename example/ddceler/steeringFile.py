"""

Minimal proof of concept for DDG4 steering file to run Geant4 with celeritas integration.

Usage:

CELER_LOG=debug CELER_LOG_LOCAL=debug ddsim --action.run DDcelerRunAction --compactFile=$PWD/install/share/CeleritasDD4hep/celeritas-dd4hep.xml --steering steeringFile.py -N 500 -G --gun.particle e- --gun.etaMax 4 --gun.etaMin 1 --gun.energy 5*GeV --gun.distribution uniform --outputFile output1.root --outputConfig.forceDD4HEP  --action.tracker "Geant4TrackerAction" --action.trackerSDTypes "tracker" --action.calorimeter "Geant4CalorimeterAction" --action.calorimeterSDTypes "calorimeter"

"""

import DDG4

# Create DDG4 kernel
kernel  = DDG4.Kernel()

# Print detector information
DDG4.importConstants(kernel.detectorDescription(), debug=False)
geant4 = DDG4.Geant4(kernel)
geant4.registerInterruptHandler()
geant4.printDetectors()

# Set up custom physics list for celeritas integration
phys = geant4.setupPhysics('FTFP_BERT')
ph = DDG4.PhysicsList(kernel, str('DDcelerTMI'))
phys.adopt(ph)
phys.dump()
