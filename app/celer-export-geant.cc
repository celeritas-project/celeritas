//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-export-geant.cc
//! Import Celeritas input data from Geant4 and serialize as ROOT.
//---------------------------------------------------------------------------//

#include <iostream>

#include "celeritas_config.h"
#include "corecel/io/Logger.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/ext/MpiCommunicator.hh"
#include "celeritas/ext/RootExporter.hh"
#include "celeritas/ext/ScopedMpiInit.hh"

#if CELERITAS_USE_JSON
#    include <fstream>
#    include <nlohmann/json.hpp>

#    include "celeritas/ext/GeantPhysicsOptionsIO.json.hh"
#endif

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
        && MpiCommunicator::comm_world().size() > 1)
    {
        CELER_LOG(critical) << "This app cannot run in parallel";
        return EXIT_FAILURE;
    }

    if (argc != 4)
    {
        // Incorrect number of arguments: print help and exit
        cout << "Usage: " << argv[0]
             << " input.gdml [options.json, -, ''] output.root" << endl;
        return 2;
    }
    std::string gdml_input_filename  = argv[1];
    std::string option_filename      = argv[2];
    std::string root_output_filename = argv[3];

    GeantPhysicsOptions options;
    if (option_filename.empty())
    {
        CELER_LOG(info) << "Using default Celeritas Geant4 options";
    }
#if CELERITAS_USE_JSON
    else if (option_filename == "-")
    {
        auto inp = nlohmann::json::parse(std::cin);
        inp.get_to(options);
        CELER_LOG(info) << "Loaded Geant4 setup options: "
                        << nlohmann::json{options}.dump();
    }
    else
    {
        std::ifstream infile(option_filename);
        if (!infile)
        {
            CELER_LOG(critical) << "Failed to open '" << option_filename << "'";
            return EXIT_FAILURE;
        }
        auto inp = nlohmann::json::parse(infile);
        inp.get_to(options);
        CELER_LOG(info) << "Loaded Geant4 setup options from "
                        << option_filename << ": "
                        << nlohmann::json{options}.dump();
    }
#else
    else
    {
        CELER_LOG(critical) << "JSON is unavailable so only default Geant4 "
                               "options are supported: use '' as the second "
                               "argument";
        return EXIT_FAILURE;
    }
#endif

    // Initialize geant4 with basic EM physics from GDML path
    try
    {
        GeantImporter import(GeantSetup(gdml_input_filename, options));
        RootExporter  export_root(root_output_filename.c_str());

        // Read data from geant, write to ROOT
        export_root(import());
    }
    catch (const RuntimeError& e)
    {
        CELER_LOG(critical) << "Runtime error: " << e.what();
        return EXIT_FAILURE;
    }
    catch (const DebugError& e)
    {
        CELER_LOG(critical) << "Assertion failure: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
