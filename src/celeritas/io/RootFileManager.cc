//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/RootFileManager.cc
//---------------------------------------------------------------------------//
#include "RootFileManager.hh"

#include <TBranch.h>
#include <TFile.h>
#include <TTree.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with ROOT filename.
 */
RootFileManager::RootFileManager(char const* filename)
{
    CELER_EXPECT(filename);
    tfile_.reset(TFile::Open(filename, "recreate"));
    CELER_VALIDATE(tfile_->IsOpen(),
                   << "ROOT file at " << filename
                   << " did not open correctly.");
}

//---------------------------------------------------------------------------//
/*!
 * Create tree by providing its name and title.
 *
 * It is still possible to simply create a `TTree("name", "title")` in any
 * scope that a RootFileManager exists, but this function explicitly shows the
 * relationship between the newly created tree and `this->tfile_`.
 *
 * To expand this class to write multiple root files (one per thread), add a
 * `tid` input parameter and call `tfile_[tid].get()`.
 */
UPRootWritable<TTree>
RootFileManager::make_tree(char const* name, char const* title)
{
    CELER_EXPECT(tfile_->IsOpen());

    int const split_level = 99;
    UPRootWritable<TTree> uptree;
    uptree.reset(new TTree(name, title, split_level, tfile_.get()));
    return uptree;
}

//---------------------------------------------------------------------------//
/*!
 * Write TFile to disk.
 */
void RootFileManager::write()
{
    CELER_EXPECT(tfile_->IsOpen());
    int write_status = tfile_->Write();
    CELER_ENSURE(write_status);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
