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
template<Ownership W>
class GeantTrackView
{
    static_assert(W != Ownership::value, "GeantTrackView cannot own data");
};

//---------------------------------------------------------------------------//
/*!
 * Access track data from Geant4 with Celeritas units.
 *
 * This provides a uniform interface to G4Track data using Celeritas types and
 * units. Geant4 data are all in double precision.
 *
 * The const_reference version provides read-only access, while the reference
 * version adds setters.
 */
template<>
class GeantTrackView<Ownership::const_reference>
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

    //! Access the G4 track directly
    G4Track const& track() const { return track_; }

  private:
    G4Track const& track_;
};

//---------------------------------------------------------------------------//
/*!
 * Mutable track view with setters.
 */
template<>
class GeantTrackView<Ownership::reference>
    : public GeantTrackView<Ownership::const_reference>
{
    using Base = GeantTrackView<Ownership::const_reference>;

  public:
    //!@{
    //! \name Type aliases
    using Energy = typename Base::Energy;
    using real_type = typename Base::real_type;
    //!@}

  public:
    // Construct from mutable G4Track
    explicit GeantTrackView(G4Track& track) : Base(track) {}

    // Bring base class getters into scope
    using Base::dir;
    using Base::energy;
    using Base::pos;
    using Base::time;
    using Base::weight;

    // Setters
    inline void pos(Real3 const& position);
    inline void dir(Real3 const& direction);
    inline void energy(Energy e);
    inline void time(real_type t);

    //! Set statistical weight
    void weight(real_type w) { this->mtrack().SetWeight(w); }

  private:
    //! Access mutable track reference (safe: constructed from non-const)
    G4Track& mtrack() { return const_cast<G4Track&>(this->track()); }
};

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

using GeantTrackViewConst = GeantTrackView<Ownership::const_reference>;
using GeantTrackViewMutable = GeantTrackView<Ownership::reference>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get particle definition view.
 */
GeantParticleView GeantTrackView<Ownership::const_reference>::particle() const
{
    CELER_EXPECT(track_.GetDefinition());
    return GeantParticleView{*track_.GetDefinition()};
}

//---------------------------------------------------------------------------//
/*!
 * Get position in native Celeritas length units.
 */
Real3 GeantTrackView<Ownership::const_reference>::pos() const
{
    return convert_from_geant(track_.GetPosition(), clhep_length);
}

//---------------------------------------------------------------------------//
/*!
 * Get momentum direction.
 */
Real3 GeantTrackView<Ownership::const_reference>::dir() const
{
    return convert_from_geant(track_.GetMomentumDirection(), 1);
}

//---------------------------------------------------------------------------//
/*!
 * Get kinetic energy in MeV.
 */
auto GeantTrackView<Ownership::const_reference>::energy() const -> Energy
{
    return Energy{convert_from_geant(track_.GetKineticEnergy(), CLHEP::MeV)};
}

//---------------------------------------------------------------------------//
/*!
 * Get global time in native Celeritas time units.
 */
real_type GeantTrackView<Ownership::const_reference>::time() const
{
    return convert_from_geant(track_.GetGlobalTime(), clhep_time);
}

//---------------------------------------------------------------------------//
/*!
 * Set position in native Celeritas length units.
 */
void GeantTrackView<Ownership::reference>::pos(Real3 const& position)
{
    this->mtrack().SetPosition(convert_to_geant(position, clhep_length));
}

//---------------------------------------------------------------------------//
/*!
 * Set momentum direction.
 */
void GeantTrackView<Ownership::reference>::dir(Real3 const& direction)
{
    this->mtrack().SetMomentumDirection(convert_to_geant(direction, 1));
}

//---------------------------------------------------------------------------//
/*!
 * Set kinetic energy in MeV.
 */
void GeantTrackView<Ownership::reference>::energy(Energy e)
{
    this->mtrack().SetKineticEnergy(convert_to_geant(e.value(), CLHEP::MeV));
}

//---------------------------------------------------------------------------//
/*!
 * Set global time in native Celeritas time units.
 */
void GeantTrackView<Ownership::reference>::time(real_type t)
{
    this->mtrack().SetGlobalTime(convert_to_geant(t, clhep_time));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
