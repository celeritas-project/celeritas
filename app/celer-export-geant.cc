//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-export-geant.cc
//! Import Celeritas input data from Geant4 and serialize as ROOT.
//---------------------------------------------------------------------------//

#include <iostream>

#include "comm/Communicator.hh"
#include "comm/Logger.hh"
#include "comm/ScopedMpiInit.hh"
#include "io/GeantImporter.hh"
#include "io/GeantSetup.hh"
#include "io/RootExporter.hh"

using namespace celeritas;
using std::cout;
using std::endl;

//---------------------------------------------------------------------------//
/*!
 * This application exports particles, processes, models, XS physics
 * tables, material, and volume information constructed by the physics list and
 * loaded by the GDML geometry.
 *
 * The data is stored into a ROOT file as an \c ImportData struct.
 */
int main(int argc, char* argv[])
{
    ScopedMpiInit scoped_mpi(&argc, &argv);
    if (ScopedMpiInit::status() == ScopedMpiInit::Status::initialized
        && Communicator::comm_world().size() > 1)
    {
        CELER_LOG(critical) << "This app cannot run in parallel";
        return EXIT_FAILURE;
    }

    if (argc != 3)
    {
        // Incorrect number of arguments: print help and exit
        cout << "Usage: " << argv[0] << " geometry.gdml output.root" << endl;
        return 2;
    }
    std::string gdml_input_filename  = argv[1];
    std::string root_output_filename = argv[2];

    // Initailize geant4 with basic EM physics from GDML path
    try
    {
        GeantImporter import(GeantSetup(gdml_input_filename,
                                        GeantSetup::PhysicsList::em_basic));
        RootExporter  export_root(root_output_filename.c_str());

        // Read data from geant, write to ROOT
        export_root(import());
    }
    catch (const DebugError& e)
    {
        CELER_LOG(critical)
            << "Exception while exporting Geant4 data to root: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
