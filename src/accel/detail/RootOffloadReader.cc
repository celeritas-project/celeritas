//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/RootOffloadReader.cc
//---------------------------------------------------------------------------//
#include "RootOffloadReader.hh"

#include <TFile.h>
#include <TLeaf.h>
#include <TTree.h>

#include "corecel/io/Logger.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with ROOT input filename.
 */
RootOffloadReader::RootOffloadReader(std::string const& filename,
                                     SPConstParticles params)
    : params_(std::move(params))
{
    CELER_EXPECT(!filename.empty());
    tfile_.reset(TFile::Open(filename.c_str(), "read"));
    CELER_ASSERT(tfile_->IsOpen());
    ttree_.reset(tfile_->Get<TTree>(tree_name()));
    CELER_ASSERT(ttree_);

    num_entries_ = ttree_->GetEntries();
    CELER_ASSERT(num_entries_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Read single event from the primaries tree.
 */
auto RootOffloadReader::operator()() -> result_type
{
    std::lock_guard scoped_lock{read_mutex_};

    CELER_EXPECT(entry_count_ <= num_entries_);
    ttree_->GetEntry(entry_count_);
    auto const this_evt_id = ttree_->GetLeaf("event_id")->GetValue();

    result_type primaries;
    for (; entry_count_ < num_entries_; entry_count_++)
    {
        ttree_->GetEntry(entry_count_);
        if (ttree_->GetLeaf("event_id")->GetValue() != this_evt_id)
        {
            break;
        }

        Primary primary;
        primary.event_id = EventId{this->from_leaf<std::size_t>("event_id")};
        primary.track_id = TrackId{this->from_leaf<std::size_t>("track_id")};
        primary.particle_id
            = params_->find(PDGNumber{this->from_leaf<int>("particle")});
        primary.energy = units::MevEnergy{this->from_leaf<double>("energy")};
        primary.time = this->from_leaf<double>("time");
        primary.position = this->from_array_leaf("pos");
        primary.direction = this->from_array_leaf("dir");
        primaries.push_back(std::move(primary));
    }

    CELER_LOG_LOCAL(info) << "Read event " << this_evt_id << " with "
                          << primaries.size() << " primaries";
    return primaries;
}

//---------------------------------------------------------------------------//
/*!
 * Helper function to fetch leaves.
 */
template<class T>
auto RootOffloadReader::from_leaf(char const* leaf_name) -> T
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
Real3 RootOffloadReader::from_array_leaf(char const* leaf_name)
{
    CELER_EXPECT(ttree_);
    auto const leaf = ttree_->GetLeaf(leaf_name);
    CELER_ASSERT(leaf);
    CELER_ASSERT(leaf->GetLen() == 3);
    return {leaf->GetValue(0), leaf->GetValue(1), leaf->GetValue(2)};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas