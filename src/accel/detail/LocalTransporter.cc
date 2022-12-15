//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/LocalTransporter.cc
//---------------------------------------------------------------------------//
#include "LocalTransporter.hh"

#include <G4SystemOfUnits.hh>

#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared (MT) params.
 */
LocalTransporter::LocalTransporter(SPConstOptions opts, SPConstParams params)
    : opts_(opts), params_(params)
{
    CELER_EXPECT(opts_);
    CELER_EXPECT(params_);

    StepperInput inp{params_, opts_->max_num_tracks, opts_->sync};
    if (celeritas::device())
    {
        step_ = std::make_shared<Stepper<MemSpace::device>>(inp);
    }
    else
    {
        step_ = std::make_shared<Stepper<MemSpace::host>>(inp);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Convert a Geant4 track to a Celeritas primary and add to buffer.
 */
void LocalTransporter::add(const G4Track& g4track)
{
    CELER_EXPECT(event_);

    Primary track;

    track.particle_id = params_->particle()->find(
        PDGNumber{g4track.GetDefinition()->GetPDGEncoding()});
    track.energy = units::MevEnergy{g4track.GetKineticEnergy() / MeV};

    G4ThreeVector pos = g4track.GetPosition();
    track.position    = Real3{pos.x() / cm, pos.y() / cm, pos.z() / cm};

    G4ThreeVector dir = g4track.GetMomentumDirection();
    track.direction   = Real3{dir.x(), dir.y(), dir.z()};

    track.time     = g4track.GetGlobalTime() / s;
    track.track_id = TrackId{TrackId::size_type(g4track.GetTrackID())};
    track.event_id = event_;

    buffer_.push_back(track);
}

//---------------------------------------------------------------------------//
/*!
 * Transport the buffered tracks and all secondaries produced.
 */
void LocalTransporter::flush()
{
    if (buffer_.empty())
    {
        return;
    }

    // Copy buffered tracks to device and transport the first step
    auto track_counts = (*step_)(make_span(buffer_));

    size_type step_iters = 1;

    while (track_counts)
    {
        CELER_VALIDATE(step_iters < opts_->max_steps,
                       << "number of step iterations exceeded the allowed "
                          "maximum ("
                       << opts_->max_steps << ")");

        track_counts = (*step_)();
        ++step_iters;
    }

    buffer_.clear();
}

//---------------------------------------------------------------------------//
/*!
 * Set the event ID at the beginning of an event.
 */
void LocalTransporter::set_event(EventId event)
{
    CELER_EXPECT(event);
    CELER_EXPECT(buffer_.empty());
    event_ = event;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
