//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootImporter.cc
//---------------------------------------------------------------------------//
#include "RootImporter.hh"

#include <TBranch.h>
#include <TFile.h>
#include <TLeaf.h>
#include <TTree.h>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/sys/ScopedMem.hh"
#include "celeritas/io/ImportData.hh"

#include "RootFileManager.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from path to ROOT file.
 */
RootImporter::RootImporter(char const* filename)
{
    CELER_LOG(info) << "Opening ROOT file at " << filename;
    CELER_VALIDATE(RootFileManager::use_root(),
                   << "cannot interface with ROOT (disabled by user "
                      "environment)");

    ScopedMem record_mem("RootImporter.open");
    ScopedTimeLog scoped_time;
    root_input_.reset(TFile::Open(filename, "read"));
    CELER_VALIDATE(root_input_ && !root_input_->IsZombie(),
                   << "failed to open ROOT file");
    CELER_ENSURE(root_input_ && !root_input_->IsZombie());
}

//---------------------------------------------------------------------------//
/*!
 * Load data from the ROOT input file.
 */
ImportData RootImporter::operator()()
{
    CELER_LOG(debug) << "Reading data from ROOT";
    ScopedMem record_mem("RootImporter.read");
    ScopedTimeLog scoped_time;

    std::unique_ptr<TTree> tree_data(root_input_->Get<TTree>(tree_name()));
    CELER_ASSERT(tree_data);
    CELER_ASSERT(tree_data->GetEntries() == 1);

    ImportData import_data;
    ImportData* import_data_ptr = &import_data;
    int err_code = tree_data->SetBranchAddress(branch_name(), &import_data_ptr);
    CELER_ASSERT(err_code >= 0);
    tree_data->GetEntry(0);

    // Convert (if necessary) the resulting data to the native unit system
    convert_to_native(&import_data);

    return import_data;
}

//---------------------------------------------------------------------------//
/*!
 * Hardcoded ROOT TTree name, consistent with \e app/celer-export-geant.
 */
char const* RootImporter::tree_name()
{
    return "geant4_data";
}

//---------------------------------------------------------------------------//
/*!
 * Hardcoded ROOT TBranch name, consistent with \e app/celer-export-geant.
 */
char const* RootImporter::branch_name()
{
    return "ImportData";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
