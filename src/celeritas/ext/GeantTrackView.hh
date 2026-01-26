//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <CLHEP/Units/SystemOfUnits.h>
#include <G4Track.hh>

#include "corecel/Types.hh"
#include "corecel/math/Quantity.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/UnitTypes.hh"

#include "GeantParticleView.hh"
#include "GeantUnits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access track data from Geant4 with Celeritas units.
 *
 * This provides a uniform interface to G4Track data using Celeritas types and
 * units. Geant4 data are all in double precision.
 */
class GeantTrackView
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = Quantity<units::Mev, double>;
    using real_type = double;
    //!@}

  public:
    // Construct from G4Track
    explicit GeantTrackView(G4Track const& track) : track_(track) {}

    // Get particle definition view
    inline GeantParticleView particle() const;

    // Position in native Celeritas length units
    inline Real3 pos() const;

    // Momentum direction (unit vector)
    inline Real3 dir() const;

    // Kinetic energy [MeV]
    inline Energy energy() const;

    // Global time in native Celeritas time units
    inline real_type time() const;

    //! Statistical weight
    real_type weight() const { return track_.GetWeight(); }

  private:
    G4Track const& track_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get particle definition view.
 */
GeantParticleView GeantTrackView::particle() const
{
    CELER_EXPECT(track_.GetDefinition());
    return GeantParticleView{*track_.GetDefinition()};
}

//---------------------------------------------------------------------------//
/*!
 * Get position in native Celeritas length units.
 */
Real3 GeantTrackView::pos() const
{
    return convert_from_geant(track_.GetPosition(), clhep_length);
}

//---------------------------------------------------------------------------//
/*!
 * Get momentum direction.
 */
Real3 GeantTrackView::direction() const
{
    return convert_from_geant(track_.GetMomentumDirection(), 1);
}

//---------------------------------------------------------------------------//
/*!
 * Get kinetic energy in MeV.
 */
GeantTrackView::Energy GeantTrackView::energy() const
{
    return Energy{convert_from_geant(track_.GetKineticEnergy(), CLHEP::MeV)};
}

//---------------------------------------------------------------------------//
/*!
 * Get global time in native Celeritas time units.
 */
real_type GeantTrackView::time() const
{
    return convert_from_geant(track_.GetGlobalTime(), clhep_time);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
