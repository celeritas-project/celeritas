//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantStepPointView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <CLHEP/Units/SystemOfUnits.h>
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
    using Energy = Quantity<units::Mev, double>;
    using real_type = double;
    //!@}

  public:
    // Construct from G4StepPoint
    explicit GeantStepPointView(G4StepPoint& step_point)
        : step_point_(step_point)
    {
    }

    //!@{
    //! \name Accessors

    // Position in native Celeritas length units
    inline Real3 pos() const;

    // Momentum direction (unit vector)
    inline Real3 dir() const;

    // Kinetic energy [MeV]
    inline Energy energy() const;

    // Global time in native Celeritas time units
    inline real_type time() const;

    //! Statistical weight
    real_type weight() const { return step_point_.GetWeight(); }

    //!@}
    //!@{
    //! \name Mutators

    // Set position in native Celeritas length units
    inline void pos(Real3 const& position);

    // Set momentum direction (unit vector)
    inline void dir(Real3 const& direction);

    // Set kinetic energy [MeV]
    inline void energy(Energy kinetic_energy);

    // Set global time in native Celeritas time units
    inline void time(real_type global_time);

    // Set statistical weight
    void weight(real_type w) { step_point_.SetWeight(w); }

    // Update attributes from logical volume
    void update_from_volume(G4LogicalVolume const& lv);

    // Update attributes from touchable's logical volume
    void update_from_volume();

    // Update mass and charge from particle definition
    void update_from_particle(GeantParticleView const& particle);

    //!@}

  private:
    G4StepPoint& step_point_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get position in native Celeritas length units.
 */
Real3 GeantStepPointView::pos() const
{
    return convert_from_geant(step_point_.GetPosition(), clhep_length);
}

//---------------------------------------------------------------------------//
/*!
 * Get momentum direction.
 */
Real3 GeantStepPointView::dir() const
{
    return convert_from_geant(step_point_.GetMomentumDirection(), 1);
}

//---------------------------------------------------------------------------//
/*!
 * Get kinetic energy in MeV.
 */
GeantStepPointView::Energy GeantStepPointView::energy() const
{
    return Energy{
        convert_from_geant(step_point_.GetKineticEnergy(), CLHEP::MeV)};
}

//---------------------------------------------------------------------------//
/*!
 * Get global time in native Celeritas time units.
 */
real_type GeantStepPointView::time() const
{
    return convert_from_geant(step_point_.GetGlobalTime(), clhep_time);
}

//---------------------------------------------------------------------------//
/*!
 * Set position in native Celeritas length units.
 */
void GeantStepPointView::pos(Real3 const& position)
{
    step_point_.SetPosition(convert_to_geant(position, clhep_length));
}

//---------------------------------------------------------------------------//
/*!
 * Set momentum direction.
 */
void GeantStepPointView::dir(Real3 const& direction)
{
    step_point_.SetMomentumDirection(convert_to_geant(direction, 1));
}

//---------------------------------------------------------------------------//
/*!
 * Set kinetic energy in MeV.
 */
void GeantStepPointView::energy(Energy kinetic_energy)
{
    CELER_EXPECT(kinetic_energy >= zero_quantity());
    step_point_.SetKineticEnergy(
        convert_to_geant(kinetic_energy.value(), CLHEP::MeV));
}

//---------------------------------------------------------------------------//
/*!
 * Set global time in native Celeritas time units.
 */
void GeantStepPointView::time(real_type global_time)
{
    CELER_EXPECT(global_time >= 0);
    step_point_.SetGlobalTime(convert_to_geant(global_time, clhep_time));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
