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
#include "GeantTrackView.hh"

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
    explicit GeantStepView(G4Step& step) : step_(step)
    {
        CELER_EXPECT(step.GetTrack());
    }

    //!@{
    //! \name Getters

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
    //! \name Setters

    // Set total energy deposited during step [MeV]
    inline void energy_deposition(Energy edep);

    // Set step length in native Celeritas length units
    inline void step_length(real_type length);

    // Update track from step data
    inline void update_track();

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
GeantStepView::Energy GeantStepView::energy_deposition() const
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
    // Set on track as well
    step_.GetTrack()->SetStepLength(step_.GetStepLength());
}

//---------------------------------------------------------------------------//
/*!
 * Get pre-step point.
 */
GeantStepPointView GeantStepView::pre_step() const
{
    return GeantStepPointView{step_.GetPreStepPoint()};
}

//---------------------------------------------------------------------------//
/*!
 * Get post-step point.
 */
GeantStepPointView GeantStepView::post_step() const
{
    return GeantStepPointView{step_.GetPostStepPoint()};
}

//---------------------------------------------------------------------------//
/*!
 * Get step point by enum.
 */
GeantStepPointView GeantStepView::step_point(StepPoint sp) const
{
    CELER_EXPECT(sp != StepPoint::size_);
    return GeantStepPointView{sp == StepPoint::pre ? this->pre_step()
                                                   : this->post_step()};
}

//---------------------------------------------------------------------------//
/*!
 * Update track from step data.
 *
 * Copies step length and step point data to the track. This is similar to
 * \c G4Step::UpdateTrack but applies only to attributes we know about and
 * safely handles null pointers.
 */
void GeantStepView::update_track()
{
    CELER_EXPECT(step_.GetTrack());

    GeantTrackView track{*step_.GetTrack()};
    GeantParticleView particle_view = track.particle();

    // Update pre-step point if present
    if (G4StepPoint* pre_step = step_.GetPreStepPoint())
    {
        GeantStepPointView{pre_step}.update_from_particle(particle_view);
        track.mtrack().SetTouchableHandle(pre_step->GetTouchableHandle());
    }

    // Update post-step point and track from post-step if present
    if (G4StepPoint* post_step = step_.GetPostStepPoint())
    {
        GeantStepPointView post_view{post_step};
        post_view.update_from_particle(particle_view);

        // Copy post-step state to track
        track.time(post_view.time());
        track.pos(post_view.pos());
        track.energy(post_view.energy());
        track.dir(post_view.dir());
        track.weight(post_view.weight());

        track.mtrack().SetNextTouchableHandle(post_step->GetTouchableHandle());
        track.mtrack().SetVelocity(post_step->GetVelocity());
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
