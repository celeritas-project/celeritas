//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/LocalTransporter.cc
//---------------------------------------------------------------------------//
#include "LocalTransporter.hh"

#include <CLHEP/Units/SystemOfUnits.h>

#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "SetupOptions.hh"
#include "SharedParams.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
template<class T>
inline T convert_from_geant(const T& val, T units)
{
    return val / units;
}

//---------------------------------------------------------------------------//
inline Real3 convert_from_geant(const G4ThreeVector& vec, double units)
{
    return {vec[0] / units, vec[1] / units, vec[2] / units};
}

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with shared (MT) params.
 */
LocalTransporter::LocalTransporter(const SetupOptions& options,
                                   const SharedParams& params)
    : auto_flush_(options.max_num_tracks), max_steps_(options.max_steps)
{
    CELER_EXPECT(params);
    particles_ = params.Params()->particle();

    StepperInput inp{params.Params(), options.max_num_tracks, options.sync};
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
 * Set the event ID at the start of an event.
 */
void LocalTransporter::SetEventId(int id)
{
    CELER_EXPECT(id >= 0);
    event_id_ = EventId(id);
    track_counter_ = 0;
}

//---------------------------------------------------------------------------//
/*!
 * Convert a Geant4 track to a Celeritas primary and add to buffer.
 */
bool LocalTransporter::TryOffload(const G4Track& g4track)
{
    CELER_EXPECT(event_id_);

    PDGNumber pdg{g4track.GetDefinition()->GetPDGEncoding()};
    if (!particles_->find(pdg))
    {
        // Celeritas doesn't know about this particle type: exit early
        return false;
    }

    Primary track;

    track.particle_id = particles_->find(
        PDGNumber{g4track.GetDefinition()->GetPDGEncoding()});
    track.energy = units::MevEnergy{
        convert_from_geant(g4track.GetKineticEnergy(), CLHEP::MeV)};

    track.position  = convert_from_geant(g4track.GetPosition(), CLHEP::cm);
    track.direction = convert_from_geant(g4track.GetMomentumDirection(), 1);
    track.time      = convert_from_geant(g4track.GetGlobalTime(), CLHEP::s);

    // TODO: Celeritas track IDs are independent from Geant4 track IDs, since
    // they must be sequential from zero for a given event. We may need to save
    // (and share with sensitive detectors!) a map of track IDs for calling
    // back to Geant4.
    track.track_id = TrackId{track_counter_++};
    track.event_id = event_id_;

    buffer_.push_back(track);
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Transport the buffered tracks and all secondaries produced.
 */
void LocalTransporter::Flush()
{
    if (buffer_.empty())
    {
        return;
    }

    CELER_LOG_LOCAL(info) << "Transporting " << buffer_.size()
                          << " tracks with Celeritas";

    // Copy buffered tracks to device and transport the first step
    auto track_counts = (*step_)(make_span(buffer_));

    size_type step_iters = 1;

    while (track_counts)
    {
        CELER_VALIDATE(step_iters < max_steps_,
                       << "number of step iterations exceeded the allowed "
                          "maximum ("
                       << max_steps_ << ")");

        track_counts = (*step_)();
        ++step_iters;
    }

    buffer_.clear();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
