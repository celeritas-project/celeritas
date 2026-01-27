//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantStepView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <CLHEP/Units/SystemOfUnits.h>
#include <G4Step.hh>

#include "corecel/Assert.hh"
#include "corecel/math/Quantity.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/Types.hh"
#include "celeritas/UnitTypes.hh"

#include "GeantStepPointView.hh"

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
    using Energy = Quantity<units::Mev, double>;
    using real_type = double;
    //!@}

  public:
    // Construct from G4Step
    explicit GeantStepView(G4Step& step) : step_(step) {}

    //!@{
    //! \name Accessors

    // Total energy deposited during step [MeV]
    inline Energy energy_deposition() const;

    // Step length in native Celeritas length units
    inline real_type step_length() const;

    // Pre-step point accessor
    inline GeantStepPointView pre_step() const;

    // Post-step point accessor
    inline GeantStepPointView post_step() const;

    // Step point accessor by enum
    inline GeantStepPointView step_point(StepPoint sp) const;

    //!@}
    //!@{
    //! \name Mutators

    // Set total energy deposited during step [MeV]
    inline void energy_deposition(Energy edep);

    // Set step length in native Celeritas length units
    inline void step_length(real_type length);

    // Update track from step data
    void update_track();

    //!@}

  private:
    G4Step& step_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get total energy deposited during step in MeV.
 */
auto GeantStepView::energy_deposition() const -> Energy
{
    return Energy{
        convert_from_geant(step_.GetTotalEnergyDeposit(), CLHEP::MeV)};
}

//---------------------------------------------------------------------------//
/*!
 * Get step length in native Celeritas length units.
 */
real_type GeantStepView::step_length() const
{
    return convert_from_geant(step_.GetStepLength(), clhep_length);
}

//---------------------------------------------------------------------------//
/*!
 * Set total energy deposited during step in MeV.
 */
void GeantStepView::energy_deposition(Energy edep)
{
    step_.SetTotalEnergyDeposit(convert_to_geant(edep.value(), CLHEP::MeV));
}

//---------------------------------------------------------------------------//
/*!
 * Set step length in native Celeritas length units.
 */
void GeantStepView::step_length(real_type length)
{
    step_.SetStepLength(convert_to_geant(length, clhep_length));
    if (step_.GetTrack())
    {
        // Set on track as well
        step_.GetTrack()->SetStepLength(step_.GetStepLength());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get pre-step point.
 */
GeantStepPointView GeantStepView::pre_step() const
{
    CELER_EXPECT(step_.GetPreStepPoint());
    return GeantStepPointView{*step_.GetPreStepPoint()};
}

//---------------------------------------------------------------------------//
/*!
 * Get post-step point.
 */
GeantStepPointView GeantStepView::post_step() const
{
    CELER_EXPECT(step_.GetPostStepPoint());
    return GeantStepPointView{*step_.GetPostStepPoint()};
}

//---------------------------------------------------------------------------//
/*!
 * Get step point by enum.
 */
GeantStepPointView GeantStepView::step_point(StepPoint sp) const
{
    CELER_EXPECT(sp != StepPoint::size_);
    return sp == StepPoint::pre ? this->pre_step() : this->post_step();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
