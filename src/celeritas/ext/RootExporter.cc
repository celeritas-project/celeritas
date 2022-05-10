//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootExporter.cc
//---------------------------------------------------------------------------//
#include "RootExporter.hh"

#include <TBranch.h>
#include <TFile.h>
#include <TLeaf.h>
#include <TTree.h>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "celeritas/io/ImportData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from path to ROOT file.
 */
RootExporter::RootExporter(const char* filename)
{
    CELER_LOG(info) << "Creating ROOT file at " << filename;
    ScopedTimeLog scoped_time;
    root_output_.reset(TFile::Open(filename, "recreate"));
    CELER_VALIDATE(root_output_ && !root_output_->IsZombie(),
                   << "failed to open ROOT file");
}

//---------------------------------------------------------------------------//
/*!
 * Write data to the ROOT file.
 */
void RootExporter::operator()(const ImportData& import_data)
{
    TTree    tree_data(tree_name(), tree_name());
    TBranch* branch = tree_data.Branch(branch_name(),
                                       const_cast<ImportData*>(&import_data));
    CELER_VALIDATE(branch, << "failed to initialize ROOT ImportData");

    // Write data to disk
    tree_data.Fill();
    int err_code = root_output_->Write();
    CELER_ENSURE(err_code >= 0);
}

//---------------------------------------------------------------------------//
/*!
 * Hardcoded ROOT TTree name, consistent with \e app/geant-exporter.
 */
const char* RootExporter::tree_name()
{
    return "geant4_data";
}

//---------------------------------------------------------------------------//
/*!
 * Hardcoded ROOT TBranch name, consistent with \e app/geant-exporter.
 */
const char* RootExporter::branch_name()
{
    return "ImportData";
}

//---------------------------------------------------------------------------//
} // namespace celeritas
