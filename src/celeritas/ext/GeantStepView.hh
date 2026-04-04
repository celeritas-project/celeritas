//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantStepView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4Step.hh>

#include "corecel/Assert.hh"
#include "corecel/math/Quantity.hh"
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
    using Energy = units::ClhepEnergy;
    using Length = lengthunits::ClhepLength;
    using real_type = double;
    //!@}

  public:
    // Construct from G4Step
    explicit GeantStepView(G4Step& step) : s_(step) {}

    //!@{
    //! \name Accessors

    // Total energy deposited during step [MeV]
    inline Energy energy_deposition() const;

    // Step length in CLHEP length units (mm)
    inline Length step_length() const;

    // Pre-step point accessor
    inline GeantStepPointView pre_step() const;

    // Post-step point accessor
    inline GeantStepPointView post_step() const;

    // Whether the step point is valid (not null)
    inline bool has_step_point(StepPoint sp) const;

    // Step point accessor by enum
    inline GeantStepPointView step_point(StepPoint sp) const;

    //!@}
    //!@{
    //! \name Mutators

    // Set total energy deposited during step [MeV]
    inline void energy_deposition(Energy edep);

    // Set step length in CLHEP length units (mm)
    inline void step_length(Length length);

    // Update track from step data
    void update_track();

    // Delete a step point if not used for SD reconstruction
    void delete_step_point(StepPoint sp);

    //!@}

  private:
    G4Step& s_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get total energy deposited during step in MeV.
 */
auto GeantStepView::energy_deposition() const -> Energy
{
    return Energy{s_.GetTotalEnergyDeposit()};
}

//---------------------------------------------------------------------------//
/*!
 * Get step length in CLHEP length units (mm).
 */
GeantStepView::Length GeantStepView::step_length() const
{
    return Length{s_.GetStepLength()};
}

//---------------------------------------------------------------------------//
/*!
 * Set total energy deposited during step in MeV.
 */
void GeantStepView::energy_deposition(Energy edep)
{
    s_.SetTotalEnergyDeposit(edep.value());
}

//---------------------------------------------------------------------------//
/*!
 * Set step length in native Celeritas length units.
 */
void GeantStepView::step_length(Length length)
{
    s_.SetStepLength(length.value());
    if (s_.GetTrack())
    {
        // Set on track as well
        s_.GetTrack()->SetStepLength(s_.GetStepLength());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get pre-step point.
 */
GeantStepPointView GeantStepView::pre_step() const
{
    CELER_EXPECT(this->has_step_point(StepPoint::pre));
    return GeantStepPointView{*s_.GetPreStepPoint()};
}

//---------------------------------------------------------------------------//
/*!
 * Get post-step point.
 */
GeantStepPointView GeantStepView::post_step() const
{
    CELER_EXPECT(this->has_step_point(StepPoint::post));
    return GeantStepPointView{*s_.GetPostStepPoint()};
}

//---------------------------------------------------------------------------//
/*!
 * Whether the step point is valid (not null).
 */
bool GeantStepView::has_step_point(StepPoint sp) const
{
    CELER_EXPECT(sp != StepPoint::size_);
    return (sp == StepPoint::pre ? s_.GetPreStepPoint() : s_.GetPostStepPoint())
           != nullptr;
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
