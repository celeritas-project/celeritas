//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantTrackOrder.cc
//---------------------------------------------------------------------------//
#include "GeantTrackOrder.hh"

#include <tuple>
#include <G4ParticleDefinition.hh>
#include <G4Track.hh>

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
auto make_key(G4Track const& t)
{
    auto const& pos = t.GetPosition();
    auto const& dir = t.GetMomentumDirection();
    return std::make_tuple(t.GetParticleDefinition()->GetPDGEncoding(),
                           t.GetKineticEnergy(),
                           t.GetGlobalTime(),
                           pos.x(),
                           pos.y(),
                           pos.z(),
                           dir.x(),
                           dir.y(),
                           dir.z(),
                           t.GetWeight(),
                           t.GetTrackID());
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Whether the left track is ordered before the right one.
 */
bool GeantTrackOrder::operator()(G4Track const& lhs, G4Track const& rhs) const
{
    return make_key(lhs) < make_key(rhs);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
