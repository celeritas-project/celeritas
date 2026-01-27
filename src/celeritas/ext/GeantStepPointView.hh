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
    explicit GeantStepPointView(G4StepPoint& step_point) : sp_(step_point) {}

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
    real_type weight() const { return sp_.GetWeight(); }

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
Real3 GeantStepPointView::pos() const
{
    return convert_from_geant(sp_.GetPosition(), clhep_length);
}

//---------------------------------------------------------------------------//
/*!
 * Get momentum direction.
 */
Real3 GeantStepPointView::dir() const
{
    return convert_from_geant(sp_.GetMomentumDirection(), 1);
}

//---------------------------------------------------------------------------//
/*!
 * Get kinetic energy in MeV.
 */
GeantStepPointView::Energy GeantStepPointView::energy() const
{
    return Energy{convert_from_geant(sp_.GetKineticEnergy(), CLHEP::MeV)};
}

//---------------------------------------------------------------------------//
/*!
 * Get global time in native Celeritas time units.
 */
real_type GeantStepPointView::time() const
{
    return convert_from_geant(sp_.GetGlobalTime(), clhep_time);
}

//---------------------------------------------------------------------------//
/*!
 * Set position in native Celeritas length units.
 */
void GeantStepPointView::pos(Real3 const& position)
{
    sp_.SetPosition(convert_to_geant(position, clhep_length));
}

//---------------------------------------------------------------------------//
/*!
 * Set momentum direction.
 */
void GeantStepPointView::dir(Real3 const& direction)
{
    sp_.SetMomentumDirection(convert_to_geant(direction, 1));
}

//---------------------------------------------------------------------------//
/*!
 * Set kinetic energy in MeV.
 */
void GeantStepPointView::energy(Energy kinetic_energy)
{
    CELER_EXPECT(kinetic_energy >= zero_quantity());
    sp_.SetKineticEnergy(convert_to_geant(kinetic_energy.value(), CLHEP::MeV));
    // TODO: update speed based on mass, KE
}

//---------------------------------------------------------------------------//
/*!
 * Set global time in native Celeritas time units.
 */
void GeantStepPointView::time(real_type global_time)
{
    CELER_EXPECT(global_time >= 0);
    sp_.SetGlobalTime(convert_to_geant(global_time, clhep_time));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
