//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/RootOffloadWriter.cc
//---------------------------------------------------------------------------//
#include "RootOffloadWriter.hh"

#include <TFile.h>
#include <TTree.h>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "orange/Types.hh"
#include "celeritas/phys/ParticleParams.hh"  // IWYU pragma: keep

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Convert celeritas::Real3 to std::array<double, 3>.
 */
std::array<double, 3> real3_to_array(Real3 const& src)
{
    std::array<double, 3> dst;
    std::memcpy(&dst, &src, sizeof(src));
    return dst;
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with ROOT output filename.
 */
RootOffloadWriter::RootOffloadWriter(std::string const& root_output_name,
                                     SPConstParticles params)
    : tfile_mgr_(root_output_name.c_str()), params_(std::move(params))
{
    ttree_ = tfile_mgr_.make_tree(this->tree_name(), this->tree_name());
    ttree_->Branch("event_id", &primary_.event_id);
    ttree_->Branch("track_id", &primary_.track_id);
    ttree_->Branch("particle", &primary_.particle);
    ttree_->Branch("energy", &primary_.energy);
    ttree_->Branch("time", &primary_.time);
    ttree_->Branch("pos", &primary_.pos);
    ttree_->Branch("dir", &primary_.dir);
}

//---------------------------------------------------------------------------//
/*!
 * Export primaries to ROOT.
 */
void RootOffloadWriter::operator()(Primaries const& primaries)
{
    CELER_EXPECT(!primaries.empty());

    std::scoped_lock{write_mutex_};
    for (auto const& p : primaries)
    {
        primary_.event_id = p.event_id.get();
        primary_.track_id = p.track_id.get();
        primary_.particle = params_->id_to_pdg(p.particle_id).get();
        primary_.energy = p.energy.value();
        primary_.time = p.time;
        primary_.pos = real3_to_array(p.position);
        primary_.dir = real3_to_array(p.direction);
        ttree_->Fill();
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
