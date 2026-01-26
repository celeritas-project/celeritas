//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantStepView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <CLHEP/Units/SystemOfUnits.h>
#include <G4Step.hh>

#include "corecel/math/Quantity.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/UnitTypes.hh"

#include "GeantUnits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access and modify step data from Geant4 with Celeritas units.
 *
 * This provides a uniform interface to G4Step data using Celeritas types and
 * units. Geant4 data are all in double precision.
 */
class GeantStepView
{
  public:
    //!@{
    //! \name Type aliases
    using MevEnergy = Quantity<units::Mev, double>;
    using Energy = MevEnergy;
    using real_type = double;
    //!@}

  public:
    // Construct from G4Step
    explicit GeantStepView(G4Step* step) : step_(step) {}

    //!@{
    //! \name Getters

    // Total energy deposited during step [MeV]
    inline MevEnergy energy_deposition() const;

    // Step length in native Celeritas length units
    inline real_type step_length() const;

    //!@}
    //!@{
    //! \name Setters

    // Set total energy deposited during step [MeV]
    inline void energy_deposition(MevEnergy edep);

    // Set step length in native Celeritas length units
    inline void step_length(real_type length);

    //!@}

  private:
    G4Step* step_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get total energy deposited during step in MeV.
 */
GeantStepView::MevEnergy GeantStepView::energy_deposition() const
{
    return MevEnergy{
        convert_from_geant(step_->GetTotalEnergyDeposit(), CLHEP::MeV)};
}

//---------------------------------------------------------------------------//
/*!
 * Get step length in native Celeritas length units.
 */
real_type GeantStepView::step_length() const
{
    return convert_from_geant(step_->GetStepLength(), clhep_length);
}

//---------------------------------------------------------------------------//
/*!
 * Set total energy deposited during step in MeV.
 */
void GeantStepView::energy_deposition(MevEnergy edep)
{
    step_->SetTotalEnergyDeposit(convert_to_geant(edep.value(), CLHEP::MeV));
}

//---------------------------------------------------------------------------//
/*!
 * Set step length in native Celeritas length units.
 */
void GeantStepView::step_length(real_type length)
{
    step_->SetStepLength(convert_to_geant(length, clhep_length));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
