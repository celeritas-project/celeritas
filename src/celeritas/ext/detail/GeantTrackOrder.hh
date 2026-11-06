//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantTrackOrder.hh
//---------------------------------------------------------------------------//
#pragma once

class G4Track;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Order Geant4 tracks by their physical state.
 *
 * This strict weak ordering compares, in order, the PDG encoding, kinetic
 * energy, global time, position, direction, and weight, with the track ID as
 * a final tiebreak.
 *
 * Tracks handed back by Celeritas are stacked in this order so that their
 * place in the Geant4 track stack (and thus in the Geant4 random number
 * sequence) does not depend on how Celeritas scheduled them: track slots,
 * compaction order, and Celeritas track IDs can all depend on the
 * (asynchronous) execution order, but the state of each track does not. The
 * track ID is only compared for bitwise-identical states.
 */
struct GeantTrackOrder
{
    // Whether the left track is ordered before the right one
    bool operator()(G4Track const& lhs, G4Track const& rhs) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
