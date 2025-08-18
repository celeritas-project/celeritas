//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
RootExporter::RootExporter(char const* filename)
{
    CELER_LOG(info) << "Creating ROOT file at " << filename;
    CELER_VALIDATE(RootFileManager::use_root(),
                   << "cannot interface with ROOT (disabled by user "
                      "environment)");

    ScopedMem record_mem("RootExporter.open");
    ScopedTimeLog scoped_time;
    root_output_.reset(TFile::Open(filename, "recreate"));
    CELER_VALIDATE(root_output_ && !root_output_->IsZombie(),
                   << "failed to open ROOT file");
}

//---------------------------------------------------------------------------//
/*!
 * Write data to the ROOT file.
 *
 * Splitting is prevented in order to reduce the file size. The default buffer
 * size (32 KB) is used.
 */
void RootExporter::operator()(ImportData const& import_data)
{
    ScopedMem record_mem("RootExporter.write");
    TTree tree_data(tree_name(), tree_name());
    TBranch* branch = tree_data.Branch(branch_name(),
                                       const_cast<ImportData*>(&import_data),
                                       /* bufsize = */ 32000,
                                       /* splitlevel = */ 0);
    CELER_VALIDATE(branch, << "failed to initialize ROOT ImportData");

    // Write data to disk
    tree_data.Fill();
    tree_data.Write();
}

//---------------------------------------------------------------------------//
/*!
 * Hardcoded ROOT TTree name, consistent with \e app/celer-export-geant.
 */
char const* RootExporter::tree_name()
{
    return "geant4_data";
}

//---------------------------------------------------------------------------//
/*!
 * Hardcoded ROOT TBranch name, consistent with \e app/celer-export-geant.
 */
char const* RootExporter::branch_name()
{
    return "ImportData";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
