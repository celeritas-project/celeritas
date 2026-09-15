//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantStepPointView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4StepPoint.hh>

#include "corecel/Assert.hh"
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
 * Access and modify step point data from Geant4 with Celeritas units.
 *
 * This provides a uniform interface to G4StepPoint data using Celeritas types
 * and units. Geant4 data are all in double precision.
 */
class GeantStepPointView
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::ClhepEnergy;
    using Length = lengthunits::ClhepLength;
    using Time = units::ClhepTime;
    using Speed = Quantity<units::CLight, double>;
    using real_type = double;
    //!@}

  public:
    // Construct from G4StepPoint
    explicit GeantStepPointView(G4StepPoint& step_point) : sp_(step_point) {}

    //!@{
    //! \name Accessors

    // Position in CLHEP length units (mm)
    inline Array<Length, 3> pos() const;

    // Momentum direction (unit vector)
    inline Array<double, 3> dir() const;

    // Kinetic energy [MeV]
    inline Energy energy() const;

    // Global time in CLHEP time units (ns)
    inline Time time() const;

    //! Statistical weight
    real_type weight() const { return sp_.GetWeight(); }

    //! Speed in units of the speed of light
    inline Speed speed() const;

    //!@}
    //!@{
    //! \name Mutators

    // Set position in CLHEP length units (mm)
    inline void pos(Array<Length, 3> const& position);

    // Set momentum direction (unit vector)
    inline void dir(Array<double, 3> const& direction);

    // Set kinetic energy [MeV]
    inline void energy(Energy kinetic_energy);

    // Set global time in CLHEP time units (ns)
    inline void time(Time global_time);

    // Set statistical weight
    void weight(real_type w) { sp_.SetWeight(w); }

    // Update attributes from logical volume
    void update_from_volume(G4LogicalVolume const& lv);

    // Update attributes from touchable's logical volume
    void update_from_volume();

    // Update mass and charge from particle definition
    void update_from_particle(GeantParticleView const& particle);

    // Clear unsupported attributes to invalid sentinel values
    void clear_unsupported();

    //!@}

    //! Access underlying G4 object
    G4StepPoint& step_point() { return sp_; }

  private:
    G4StepPoint& sp_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get position in native Celeritas length units.
 */
Array<GeantStepPointView::Length, 3> GeantStepPointView::pos() const
{
    return make_quantity_array<Length>(to_array(sp_.GetPosition()));
}

//---------------------------------------------------------------------------//
/*!
 * Get momentum direction.
 */
Array<double, 3> GeantStepPointView::dir() const
{
    return to_array(sp_.GetMomentumDirection());
}

//---------------------------------------------------------------------------//
/*!
 * Get kinetic energy in MeV.
 */
GeantStepPointView::Energy GeantStepPointView::energy() const
{
    return Energy{sp_.GetKineticEnergy()};
}

//---------------------------------------------------------------------------//
/*!
 * Get global time in native Celeritas time units.
 */
GeantStepPointView::Time GeantStepPointView::time() const
{
    return Time{sp_.GetGlobalTime()};
}

//---------------------------------------------------------------------------//
/*!
 * Get speed as a fraction of the speed of light.
 *
 * Often denoted as a particle's beta speed.
 */
GeantStepPointView::Speed GeantStepPointView::speed() const
{
    return Speed{sp_.GetBeta()};
}

//---------------------------------------------------------------------------//
/*!
 * Set position in native Celeritas length units.
 */
void GeantStepPointView::pos(Array<Length, 3> const& position)
{
    sp_.SetPosition(to_g4vector(value_as<Length>(position)));
}

//---------------------------------------------------------------------------//
/*!
 * Set momentum direction.
 */
void GeantStepPointView::dir(Array<double, 3> const& direction)
{
    sp_.SetMomentumDirection(to_g4vector(direction));
}

//---------------------------------------------------------------------------//
/*!
 * Set kinetic energy in MeV.
 */
void GeantStepPointView::energy(Energy kinetic_energy)
{
    CELER_EXPECT(kinetic_energy >= zero_quantity());
    sp_.SetKineticEnergy(kinetic_energy.value());
    // TODO: update speed based on mass, KE
}

//---------------------------------------------------------------------------//
/*!
 * Set global time in native Celeritas time units.
 */
void GeantStepPointView::time(Time global_time)
{
    sp_.SetGlobalTime(global_time.value());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
