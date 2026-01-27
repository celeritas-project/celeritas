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
 *
 * The const_reference version provides read-only access, while the reference
 * version adds setters.
 */
template<Ownership W>
class GeantTrackView
{
    static_assert(W != Ownership::value, "GeantTrackView cannot own data");
};

//---------------------------------------------------------------------------//

using GeantTrackViewConst = GeantTrackView<Ownership::const_reference>;
using GeantTrackViewMutable = GeantTrackView<Ownership::reference>;

//---------------------------------------------------------------------------//
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
    explicit GeantTrackView(G4Track const& track) : t_(track) {}

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
    real_type weight() const { return t_.GetWeight(); }

    //! Access the G4 track directly
    G4Track const& track() const { return t_; }

    //! Access the G4 track directly (const)
    G4Track const& ctrack() const { return t_; }

  private:
    G4Track const& t_;
};

//---------------------------------------------------------------------------//
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

    // Bring base class accessors into scope
    using Base::dir;
    using Base::energy;
    using Base::pos;
    using Base::time;
    using Base::weight;

    // Mutators
    inline void pos(Real3 const& position);
    inline void dir(Real3 const& direction);
    inline void energy(Energy e);
    inline void time(real_type t);

    //! Set statistical weight
    void weight(real_type w) { this->track().SetWeight(w); }

    using Base::ctrack;
    using Base::track;
    //! Access mutable track reference (safe: constructed from non-const)
    G4Track& track() { return const_cast<G4Track&>(this->ctrack()); }
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//

// Deduce const_reference from const G4Track&
GeantTrackView(G4Track const&) -> GeantTrackView<Ownership::const_reference>;

// Deduce reference from mutable G4Track&
GeantTrackView(G4Track&) -> GeantTrackView<Ownership::reference>;

// Doxygen fails to deduce correct templated class
#if !defined(__DOXYGEN__) || __DOXYGEN__ > 0x011600
//---------------------------------------------------------------------------//
// INLINE ACCESSOR DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get particle definition view.
 */
GeantParticleView GeantTrackView<Ownership::const_reference>::particle() const
{
    CELER_EXPECT(t_.GetDefinition());
    return GeantParticleView{*t_.GetDefinition()};
}

//---------------------------------------------------------------------------//
/*!
 * Get position in native Celeritas length units.
 */
Real3 GeantTrackView<Ownership::const_reference>::pos() const
{
    return convert_from_geant(t_.GetPosition(), clhep_length);
}

//---------------------------------------------------------------------------//
/*!
 * Get momentum direction.
 */
Real3 GeantTrackView<Ownership::const_reference>::dir() const
{
    return convert_from_geant(t_.GetMomentumDirection(), 1);
}

//---------------------------------------------------------------------------//
/*!
 * Get kinetic energy in MeV.
 */
auto GeantTrackView<Ownership::const_reference>::energy() const -> Energy
{
    return Energy{convert_from_geant(t_.GetKineticEnergy(), CLHEP::MeV)};
}

//---------------------------------------------------------------------------//
/*!
 * Get global time in native Celeritas time units.
 */
real_type GeantTrackView<Ownership::const_reference>::time() const
{
    return convert_from_geant(t_.GetGlobalTime(), clhep_time);
}

//---------------------------------------------------------------------------//
// INLINE MUTATOR DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Set position in native Celeritas length units.
 */
void GeantTrackView<Ownership::reference>::pos(Real3 const& position)
{
    this->track().SetPosition(convert_to_geant(position, clhep_length));
}

//---------------------------------------------------------------------------//
/*!
 * Set momentum direction.
 */
void GeantTrackView<Ownership::reference>::dir(Real3 const& direction)
{
    this->track().SetMomentumDirection(convert_to_geant(direction, 1));
}

//---------------------------------------------------------------------------//
/*!
 * Set kinetic energy in MeV.
 */
void GeantTrackView<Ownership::reference>::energy(Energy e)
{
    this->track().SetKineticEnergy(convert_to_geant(e.value(), CLHEP::MeV));
}

//---------------------------------------------------------------------------//
/*!
 * Set global time in native Celeritas time units.
 */
void GeantTrackView<Ownership::reference>::time(real_type t)
{
    this->track().SetGlobalTime(convert_to_geant(t, clhep_time));
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
