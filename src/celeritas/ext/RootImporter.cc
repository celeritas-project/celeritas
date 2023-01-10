//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
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
#include "celeritas/io/ImportData.hh"

// This "public API" function is defined in CeleritasRootInterface.cxx to
// initialize ROOT. It's not necessary for shared libraries (due to static
// initialization shenanigans) but is needed for static libs. The name is a
// function of the name passed to the MODULE argument of
// the cmake root_generate_dictionary command.
void TriggerDictionaryInitialization_libceleritas();

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from path to ROOT file.
 */
RootImporter::RootImporter(char const* filename)
{
    TriggerDictionaryInitialization_libceleritas();

    CELER_LOG(info) << "Opening ROOT file at " << filename;
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
    ScopedTimeLog scoped_time;

    std::unique_ptr<TTree> tree_data(root_input_->Get<TTree>(tree_name()));
    CELER_ASSERT(tree_data);
    CELER_ASSERT(tree_data->GetEntries() == 1);

    ImportData import_data;
    ImportData* import_data_ptr = &import_data;
    int err_code = tree_data->SetBranchAddress(branch_name(), &import_data_ptr);
    CELER_ASSERT(err_code >= 0);
    tree_data->GetEntry(0);

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
