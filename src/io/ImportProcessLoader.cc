//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportProcessLoader.cc
//---------------------------------------------------------------------------//
#include "ImportProcessLoader.hh"

#include <TFile.h>
#include <TTree.h>

#include "base/Range.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with RootLoader.
 */
ImportProcessLoader::ImportProcessLoader(RootLoader& root_loader)
    : root_loader_(root_loader)
{
    CELER_ENSURE(root_loader_);
}

//---------------------------------------------------------------------------//
/*!
 * Load ImportProcess data.
 */
const std::vector<ImportProcess> ImportProcessLoader::operator()()
{
    const auto tfile = root_loader_.get();

    std::unique_ptr<TTree> tree_processes(tfile->Get<TTree>("processes"));
    CELER_ASSERT(tree_processes);
    CELER_ASSERT(tree_processes->GetEntries());

    // Load branch
    ImportProcess  process;
    ImportProcess* process_ptr = &process;

    int err_code
        = tree_processes->SetBranchAddress("ImportProcess", &process_ptr);
    CELER_ASSERT(err_code >= 0);

    std::vector<ImportProcess> processes;

    // Populate physics process vector
    for (size_type i : range(tree_processes->GetEntries()))
    {
        tree_processes->GetEntry(i);
        processes.push_back(process);
    }

    return processes;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
