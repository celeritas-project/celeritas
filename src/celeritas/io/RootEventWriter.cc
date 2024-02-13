//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/RootEventWriter.cc
//---------------------------------------------------------------------------//
#include "RootEventWriter.hh"

#include <algorithm>
#include <set>
#include <TFile.h>
#include <TTree.h>

#include "corecel/Assert.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "geocel/Types.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/phys/ParticleParams.hh"  // IWYU pragma: keep

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Convert celeritas::Real3 to std::array<double, 3>.
 */
std::array<double, 3> real3_to_array(Real3 const& src)
{
    std::array<double, 3> dst;
    std::copy(src.begin(), src.end(), dst.begin());
    return dst;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with ROOT output filename.
 */
RootEventWriter::RootEventWriter(SPRootFileManager root_file_manager,
                                 SPConstParticles params)
    : tfile_mgr_(std::move(root_file_manager))
    , params_(std::move(params))
    , event_id_(static_cast<size_type>(-1))
{
    CELER_EXPECT(tfile_mgr_);
    CELER_EXPECT(params_);

    ScopedRootErrorHandler scoped_root_error;

    CELER_LOG(info) << "Creating event tree '" << this->tree_name() << "' at "
                    << tfile_mgr_->filename();

    ttree_ = tfile_mgr_->make_tree(this->tree_name(), this->tree_name());
    ttree_->Branch("event_id", &primary_.event_id);
    ttree_->Branch("particle", &primary_.particle);
    ttree_->Branch("energy", &primary_.energy);
    ttree_->Branch("time", &primary_.time);
    ttree_->Branch("pos", &primary_.pos);
    ttree_->Branch("dir", &primary_.dir);

    scoped_root_error.throw_if_errors();
}

//---------------------------------------------------------------------------//
/*!
 * Export primaries to ROOT.
 */
void RootEventWriter::operator()(VecPrimary const& primaries)
{
    CELER_EXPECT(!primaries.empty());
    ScopedRootErrorHandler scoped_root_error;

    // Increment contiguous event id
    event_id_++;

    std::set<EventId::size_type> mismatched_events;
    for (auto const& p : primaries)
    {
        if (p.event_id.get() != event_id_)
        {
            mismatched_events.insert(p.event_id.unchecked_get());
        }

        primary_.event_id = event_id_;
        primary_.particle = params_->id_to_pdg(p.particle_id).get();
        primary_.energy = p.energy.value();
        primary_.time = p.time;
        primary_.pos = real3_to_array(p.position);
        primary_.dir = real3_to_array(p.direction);
        ttree_->Fill();
    }

    if (!mismatched_events.empty())
    {
        CELER_LOG_LOCAL(warning)
            << "Overwriting primary event IDs with " << event_id_ << ": "
            << join(mismatched_events.begin(), mismatched_events.end(), ", ");
    }

    scoped_root_error.throw_if_errors();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
