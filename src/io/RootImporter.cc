//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RootImporter.cc
//---------------------------------------------------------------------------//
#include "RootImporter.hh"

#include <cstdlib>
#include <iomanip>
#include <tuple>

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TLeaf.h>

#include "base/Assert.hh"
#include "base/Range.hh"
#include "comm/Logger.hh"
#include "physics/base/Units.hh"
#include "detail/ImportParticle.hh"
#include "ImportData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from path to ROOT file.
 */
RootImporter::RootImporter(const char* filename)
{
    CELER_LOG(status) << "Opening ROOT file";
    root_input_.reset(TFile::Open(filename, "read"));
    CELER_ENSURE(root_input_ && !root_input_->IsZombie());
}

//---------------------------------------------------------------------------//
/*!
 * Default destructor.
 */
RootImporter::~RootImporter() = default;

//---------------------------------------------------------------------------//
/*!
 * Load data from the ROOT input file by providing the tree and branch names.
 */
ImportData
RootImporter::operator()(const char* tree_name, const char* branch_name)
{
    CELER_EXPECT(tree_name);
    CELER_EXPECT(branch_name);

    std::unique_ptr<TTree> tree_data(root_input_->Get<TTree>(tree_name));
    CELER_ASSERT(tree_data);
    CELER_ASSERT(tree_data->GetEntries() == 1);

    ImportData  import_data;
    ImportData* import_data_ptr = &import_data;
    int err_code = tree_data->SetBranchAddress(branch_name, &import_data_ptr);
    CELER_ASSERT(err_code >= 0);
    tree_data->GetEntry(0);

    return import_data;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
