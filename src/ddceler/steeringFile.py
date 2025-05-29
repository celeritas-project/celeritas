"""

Minimal proof of concept for DDG4 steering file to run Geant4 with celeritas integration.

// Build celeritas-DDG4 plugins and set library path
cmake -DCELERITAS_USE_DD4hep=ON -DCMAKE_INSTALL_PREFIX=install -S . -B build
cmake --build build -j 8
cmake --install build
export LD_LIBRARY_PATH=$PWD/install/lib:${LD_LIBRARY_PATH}

// Download and install the Open Data Detector For Testing.
git clone https://gitlab.cern.ch/acts/OpenDataDetector
cd OpenDataDetector
cmake -DCMAKE_INSTALL_PREFIX=install -S . -B build
cmake --build build -j 8
cmake --install build
source install/bin/this_odd.sh

// Run simulation with celeritas integration
CELER_LOG=debug CELER_LOG_LOCAL=debug ddsim --action.run DDcelerRunAction \
--compactFile=$PWD/install/share/OpenDataDetector/xml/OpenDataDetector.xml \
--steering steeringFile.py -N 10 -G

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
