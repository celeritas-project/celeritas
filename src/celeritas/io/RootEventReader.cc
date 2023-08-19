//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/RootEventReader.cc
//---------------------------------------------------------------------------//
#include "RootEventReader.hh"

#include <TFile.h>
#include <TLeaf.h>
#include <TTree.h>

#include "corecel/io/Logger.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with ROOT input filename.
 */
RootEventReader::RootEventReader(std::string const& filename,
                                 SPConstParticles params)
    : params_(std::move(params))
{
    ScopedRootErrorHandler scoped_root_error;

    CELER_EXPECT(!filename.empty());
    tfile_.reset(TFile::Open(filename.c_str(), "read"));
    CELER_ASSERT(tfile_->IsOpen());
    ttree_.reset(tfile_->Get<TTree>(tree_name()));
    CELER_ASSERT(ttree_);
    num_entries_ = ttree_->GetEntries();
    CELER_ASSERT(num_entries_ > 0);

    scoped_root_error.throw_if_errors();
}

//---------------------------------------------------------------------------//
/*!
 * Read single event from the primaries tree.
 */
auto RootEventReader::operator()() -> result_type
{
    ScopedRootErrorHandler scoped_root_error;

    CELER_EXPECT(entry_count_ <= num_entries_);
    size_type this_evt_id;
    TrackId track_id{0};
    result_type primaries;

    for (; entry_count_ < num_entries_; entry_count_++)
    {
        ttree_->GetEntry(entry_count_);

        if (primaries.empty())
        {
            // First entry, define event id
            this_evt_id = this->from_leaf<size_type>("event_id");
        }

        if (this->from_leaf<size_type>("event_id") != this_evt_id)
        {
            // End of primaries in this event; stop
            break;
        }

        Primary primary;
        primary.track_id = track_id;
        primary.event_id = EventId{this->from_leaf<size_type>("event_id")};
        primary.particle_id
            = params_->find(PDGNumber{this->from_leaf<int>("particle")});
        primary.energy
            = units::MevEnergy{this->from_leaf<real_type>("energy")};
        primary.time = this->from_leaf<real_type>("time");
        primary.position = this->from_array_leaf("pos");
        primary.direction = this->from_array_leaf("dir");
        primaries.push_back(std::move(primary));

        track_id++;
    }

    scoped_root_error.throw_if_errors();
    CELER_LOG_LOCAL(debug) << "Read event " << this_evt_id << " with "
                           << primaries.size() << " primaries";
    return primaries;
}

//---------------------------------------------------------------------------//
/*!
 * Helper function to fetch leaves.
 */
template<class T>
auto RootEventReader::from_leaf(char const* leaf_name) -> T
{
    CELER_EXPECT(ttree_);
    auto const leaf = ttree_->GetLeaf(leaf_name);
    CELER_ASSERT(leaf);
    return static_cast<T>(leaf->GetValue());
}

//---------------------------------------------------------------------------//
/*!
 * Helper function to fetch leaves containing `std::array<double, 3>`.
 */
Real3 RootEventReader::from_array_leaf(char const* leaf_name)
{
    CELER_EXPECT(ttree_);
    auto const leaf = ttree_->GetLeaf(leaf_name);
    CELER_ASSERT(leaf);
    CELER_ASSERT(leaf->GetLen() == 3);
    return {leaf->GetValue(0), leaf->GetValue(1), leaf->GetValue(2)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas