//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/RootEventReader.cc
//---------------------------------------------------------------------------//
#include "RootEventReader.hh"

#include <TFile.h>
#include <TTree.h>

#include "corecel/io/Logger.hh"
#include "celeritas/ext/Convert.root.hh"
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
    ttree_.reset(tfile_->Get<TTree>(this->tree_name()));
    CELER_VALIDATE(ttree_,
                   << "TTree '" << this->tree_name()
                   << "' not found. Verify that '" << filename
                   << "' is a valid input file with Celeritas primary "
                      "offloaded data");
    num_entries_ = ttree_->GetEntries();
    CELER_ASSERT(num_entries_ > 0);

    // Get the number of events. Event IDs are sequential starting from zero.
    // The last entry will contain the largest event ID.
    ttree_->GetEntry(num_entries_ - 1);
    num_events_ = from_leaf<size_type>(*ttree_->GetLeaf("event_id")) + 1;
    CELER_LOG(debug) << "ROOT file has " << num_events_ << " events";

    scoped_root_error.throw_if_errors();
}

//---------------------------------------------------------------------------//
/*!
 * Read a specific single event from the primaries tree.
 */
auto RootEventReader::operator()(EventId event_id) -> result_type
{
    CELER_EXPECT(event_id <= num_entries_);

    if (!event_to_entry_.empty())
    {
        if (event_id < event_to_entry_.size())
        {
            // Cached event entry; Load entry and return event
            entry_count_ = event_to_entry_[event_id.get()];
            return this->operator()();
        }
        else
        {
            // Continue from latest entry count
            entry_count_ = event_to_entry_.back();
        }
    }

    ScopedRootErrorHandler scoped_root_error;

    // Enable only event_id branch
    ttree_->SetBranchStatus("*", false);  // Disable all branches
    ttree_->SetBranchStatus("event_id", true);
    ttree_->GetEntry(entry_count_);
    auto expected_evt_id
        = EventId{from_leaf<size_type>(*ttree_->GetLeaf("event_id"))};

    result_type result;
    for (; entry_count_ < num_entries_; entry_count_++)
    {
        ttree_->GetEntry(entry_count_);
        auto entry_evt_id
            = EventId{from_leaf<size_type>(*ttree_->GetLeaf("event_id"))};

        if (entry_evt_id != expected_evt_id)
        {
            // Found new event; Add its entry to the cache
            event_to_entry_.push_back(entry_count_);
            expected_evt_id = entry_evt_id;
        }
        if (entry_evt_id == event_id)
        {
            // Found requested event
            result = this->operator()();
            break;
        }
    }
    scoped_root_error.throw_if_errors();

    CELER_ENSURE(!result.empty());
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Read single event from the primaries tree.
 */
auto RootEventReader::operator()() -> result_type
{
    CELER_EXPECT(entry_count_ <= num_entries_);

    EventId expected_evt_id{0};
    TrackId track_id{0};
    result_type primaries;
    ScopedRootErrorHandler scoped_root_error;

    ttree_->SetBranchStatus("*", true);  // Enable all branches
    for (; entry_count_ < num_entries_; entry_count_++)
    {
        ttree_->GetEntry(entry_count_);

        auto entry_evt_id
            = EventId{from_leaf<size_type>(*ttree_->GetLeaf("event_id"))};
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
        primary.particle_id = params_->find(
            PDGNumber{from_leaf<int>(*ttree_->GetLeaf("particle"))});
        primary.energy = units::MevEnergy{
            from_leaf<real_type>(*ttree_->GetLeaf("energy"))};
        primary.time = from_leaf<real_type>(*ttree_->GetLeaf("time"));
        primary.position = from_array_leaf(*ttree_->GetLeaf("pos"));
        primary.direction = from_array_leaf(*ttree_->GetLeaf("dir"));
        primaries.push_back(std::move(primary));

        track_id++;
    }

    scoped_root_error.throw_if_errors();
    CELER_LOG_LOCAL(debug) << "Read event " << expected_evt_id.get()
                           << " with " << primaries.size() << " primaries";
    return primaries;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
