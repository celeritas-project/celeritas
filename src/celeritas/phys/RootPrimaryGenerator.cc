//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/RootPrimaryGenerator.cc
//---------------------------------------------------------------------------//
#include "RootPrimaryGenerator.hh"

#include <TFile.h>
#include <TLeaf.h>
#include <TTree.h>

#include "corecel/Macros.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from input file, number of events, and primaries to be sampled per
 * event.
 */
RootPrimaryGenerator::RootPrimaryGenerator(std::string const& filename,
                                           SPConstParticles params,
                                           size_type num_events,
                                           size_type primaries_per_event,
                                           unsigned int seed)
    : params_(std::move(params))
    , num_events_{num_events}
    , primaries_per_event_(primaries_per_event)
{
    CELER_EXPECT(!filename.empty());
    CELER_EXPECT(num_events > 0 && primaries_per_event > 0);

    ScopedRootErrorHandler scoped_root_error;
    tfile_.reset(TFile::Open(filename.c_str(), "read"));
    CELER_ENSURE(tfile_->IsOpen());

    ttree_.reset(tfile_->Get<TTree>("primaries"));
    CELER_ENSURE(ttree_);
    CELER_VALIDATE(ttree_->GetEntries() > 0,
                   << "TTree `primaries` in '" << tfile_->GetName()
                   << "' has zero entries");
    scoped_root_error.throw_if_errors();

    entry_selector_ = std::uniform_int_distribution<>(0, primaries_per_event);
    rng_.seed(seed);
}

//---------------------------------------------------------------------------//
/*!
 * Generate primaries from ROOT input file.
 */
auto RootPrimaryGenerator::operator()() -> result_type
{
    if (event_count_.get() > num_events_)
    {
        return result_type{};
    }

    ScopedRootErrorHandler scoped_root_error;

    // Fetch ROOT leaves
    // TODO: Make these private members?
    auto lpid = ttree_->GetLeaf("particle");
    CELER_ASSERT(lpid);
    auto lenergy = ttree_->GetLeaf("energy");
    CELER_ASSERT(lenergy);
    auto lpos = ttree_->GetLeaf("pos");
    CELER_ASSERT(lpos);
    auto ldir = ttree_->GetLeaf("dir");
    CELER_ASSERT(ldir);
    auto ltime = ttree_->GetLeaf("time");
    CELER_ASSERT(ltime);

    TrackId track_id{0};
    result_type result(primaries_per_event_);
    for (auto i : range(primaries_per_event_))
    {
        ttree_->GetEntry(entry_selector_(rng_));

        result[i].particle_id
            = params_->find(PDGNumber{static_cast<int>(lpid->GetValue())});
        CELER_ENSURE(result[i].particle_id);
        result[i].energy = units::MevEnergy{lenergy->GetValue()};
        result[i].position = this->from_array_leaf(*lpos);
        result[i].direction = this->from_array_leaf(*ldir);
        result[i].time = ltime->GetValue();
        result[i].track_id = track_id;
        result[i].event_id = event_count_;

        track_id++;
    }
    event_count_++;
    scoped_root_error.throw_if_errors();

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Return a Real3 from leaves containing `std::array<double, 3>`.
 */
Real3 RootPrimaryGenerator::from_array_leaf(TLeaf const& leaf)
{
    CELER_EXPECT(leaf.GetLen() == 3);
    return {static_cast<real_type>(leaf.GetValue(0)),
            static_cast<real_type>(leaf.GetValue(1)),
            static_cast<real_type>(leaf.GetValue(2))};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
