//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
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
    CELER_EXPECT(!filename.empty());
    ScopedRootErrorHandler scoped_root_error;

    tfile_.reset(TFile::Open(filename.c_str(), "read"));
    CELER_ASSERT(tfile_->IsOpen());
    ttree_.reset(tfile_->Get<TTree>(tree_name()));
    CELER_ASSERT(ttree_);
    num_entries_ = ttree_->GetEntries();
    CELER_ASSERT(num_entries_ > 0);

    // Get the number of events. Event IDs are sequential starting from zero.
    // The last entry will contain the largest event ID.
    ttree_->GetEntry(num_entries_ - 1);
    num_events_ = this->from_leaf<size_type>("event_id") + 1;
    CELER_LOG(debug) << "ROOT file has " << num_events_ << " events";

    scoped_root_error.throw_if_errors();
}

//---------------------------------------------------------------------------//
/*!
 * Read single event from the primaries tree.
 */
auto RootEventReader::operator()() -> result_type
{
    CELER_EXPECT(entry_count_ <= num_entries_);
    ScopedRootErrorHandler scoped_root_error;

    EventId expected_evt_id{0};
    TrackId track_id{0};
    result_type primaries;

    for (; entry_count_ < num_entries_; entry_count_++)
    {
        ttree_->GetEntry(entry_count_);

        auto entry_evt_id = EventId{this->from_leaf<size_type>("event_id")};
        if (primaries.empty())
        {
            // First entry; set current event id
            expected_evt_id = entry_evt_id;
        }
        if (entry_evt_id != expected_evt_id)
        {
            // End of primaries in this event
            break;
        }

        Primary primary;
        primary.track_id = track_id;
        primary.event_id = expected_evt_id;
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
    CELER_LOG_LOCAL(debug) << "Read event " << expected_evt_id.get()
                           << " with " << primaries.size() << " primaries";
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
    return {static_cast<real_type>(leaf->GetValue(0)),
            static_cast<real_type>(leaf->GetValue(1)),
            static_cast<real_type>(leaf->GetValue(2))};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
