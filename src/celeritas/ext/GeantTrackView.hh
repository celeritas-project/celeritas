//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4Track.hh>

#include "corecel/Types.hh"
#include "corecel/math/Quantity.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/UnitTypes.hh"

#include "GeantParticleView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access track data from Geant4 with Celeritas units and precision.
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
    using real_type = double;
    using Energy = units::ClhepEnergy;
    using Length = lengthunits::ClhepLength;
    using Time = units::ClhepTime;
    using Real3 = Array<double, 3>;
    using ClhepLength3 = Array<Length, 3>;
    //!@}

  public:
    // Construct from G4Track
    explicit GeantTrackView(G4Track const& track) : t_(track) {}

    // Get particle definition view
    inline GeantParticleView particle() const;

    // Position in CLHEP length units (mm)
    inline auto pos() const -> ClhepLength3;

    // Momentum direction (unit vector)
    inline auto dir() const -> Real3;

    // Kinetic energy [MeV]
    inline Energy energy() const;

    // Global time in CLHEP time units (ns)
    inline Time time() const;

    //! Statistical weight
    real_type weight() const { return this->track().GetWeight(); }

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
    using Length = typename Base::Length;
    using Time = typename Base::Time;
    using Real3 = typename Base::Real3;
    using ClhepLength3 = typename Base::ClhepLength3;
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
    inline void pos(ClhepLength3 const& position);
    inline void dir(Real3 const& direction);
    inline void energy(Energy e);
    inline void time(Time t);

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
    CELER_EXPECT(this->track().GetDefinition());
    return GeantParticleView{*this->track().GetDefinition()};
}

//---------------------------------------------------------------------------//
/*!
 * Get position in CLHEP length units (mm).
 */
auto GeantTrackView<Ownership::const_reference>::pos() const -> ClhepLength3
{
    return make_quantity_array<Length>(to_array(this->track().GetPosition()));
}

//---------------------------------------------------------------------------//
/*!
 * Get momentum direction.
 */
auto GeantTrackView<Ownership::const_reference>::dir() const -> Real3
{
    return to_array(this->track().GetMomentumDirection());
}

//---------------------------------------------------------------------------//
/*!
 * Get kinetic energy in MeV.
 */
auto GeantTrackView<Ownership::const_reference>::energy() const -> Energy
{
    return Energy{this->track().GetKineticEnergy()};
}

//---------------------------------------------------------------------------//
/*!
 * Get global time in CLHEP time units (ns).
 */
auto GeantTrackView<Ownership::const_reference>::time() const -> Time
{
    return Time{this->track().GetGlobalTime()};
}

//---------------------------------------------------------------------------//
// INLINE MUTATOR DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Set position in CLHEP length units (mm).
 */
void GeantTrackView<Ownership::reference>::pos(ClhepLength3 const& position)
{
    this->track().SetPosition(to_g4vector(value_as<Length>(position)));
}

//---------------------------------------------------------------------------//
/*!
 * Set momentum direction.
 */
void GeantTrackView<Ownership::reference>::dir(Real3 const& direction)
{
    this->track().SetMomentumDirection(to_g4vector(direction));
}

//---------------------------------------------------------------------------//
/*!
 * Set kinetic energy in MeV.
 */
void GeantTrackView<Ownership::reference>::energy(Energy e)
{
    this->track().SetKineticEnergy(e.value());
}

//---------------------------------------------------------------------------//
/*!
 * Set global time in CLHEP time units (ns).
 */
void GeantTrackView<Ownership::reference>::time(Time t)
{
    this->track().SetGlobalTime(t.value());
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
