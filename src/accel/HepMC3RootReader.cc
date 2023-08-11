//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/HepMC3RootReader.cc
//---------------------------------------------------------------------------//
#include "HepMC3RootReader.hh"

#include <TFile.h>
#include <TLeaf.h>
#include <TTree.h>

#include "corecel/io/Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with ROOT input filename.
 */
HepMC3RootReader::HepMC3RootReader(std::string const& filename)
{
    CELER_EXPECT(!filename.empty());
    tfile_.reset(TFile::Open(filename.c_str(), "read"));
    CELER_ENSURE(tfile_->IsOpen());
    ttree_.reset(tfile_->Get<TTree>(tree_name()));
    CELER_ENSURE(ttree_);
    num_entries_ = ttree_->GetEntries();
    CELER_ENSURE(num_entries_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Read single entry from the primaries tree.
 */
HepMC3RootPrimary HepMC3RootReader::operator()()
{
    std::lock_guard scoped_lock{read_mutex_};

    CELER_VALIDATE(entry_count_ < num_entries_,
                   << "No more entries available for \'" << tfile_->GetName()
                   << "\'.");

    ttree_->GetEntry(entry_count_);
    entry_count_++;

    HepMC3RootPrimary primary;
    primary.event_id = ttree_->GetLeaf("event_id")->GetValue();
    primary.particle = ttree_->GetLeaf("particle")->GetValue();
    primary.energy = ttree_->GetLeaf("energy")->GetValue();
    primary.time = ttree_->GetLeaf("time")->GetValue();
    primary.pos = this->from_leaf("pos");
    primary.dir = this->from_leaf("dir");

    CELER_LOG_LOCAL(info) << "Read event " << primary.event_id;
    return primary;
}

//---------------------------------------------------------------------------//
/*!
 * Helper function to fetch an array from a TLeaf.
 */

std::array<double, 3> HepMC3RootReader::from_leaf(char const* leaf_name)
{
    CELER_EXPECT(ttree_);
    auto const leaf = ttree_->GetLeaf(leaf_name);
    CELER_ASSERT(leaf);
    CELER_ASSERT(leaf->GetLen() == 3);

    return {leaf->GetValue(0), leaf->GetValue(1), leaf->GetValue(2)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
