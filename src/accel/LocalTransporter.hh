//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/LocalTransporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4Track.hh>

#include "corecel/Types.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/Stepper.hh"

namespace celeritas
{
struct SetupOptions;
class SharedParams;
//---------------------------------------------------------------------------//
/*!
 * Manage offloading of tracks to Celeritas.
 *
 * This class must be constructed locally on each worker thread/task/stream,
 * usually as a shared pointer that's accessible to:
 * - a run action (for initialization),
 * - an event action (to set the event ID and flush offloaded tracks at the end
 *   of the event)
 * - a tracking action (to try offloading every track)
 */
class LocalTransporter
{
  public:
    // Construct in an invalid state
    LocalTransporter() = default;

    // Initialized with shared (across threads) params
    LocalTransporter(const SetupOptions& options, const SharedParams& params);

    // Set the event ID
    void SetEventId(int);

    // Whether Celeritas supports offloading of this track
    bool IsApplicable(const G4Track&) const;

    // Offload this track
    void Push(const G4Track&);

    // Transport all buffered tracks to completion
    void Flush();

    // Number of buffered tracks
    size_type GetBufferSize() const { return buffer_.size(); }

    //! Whether the class instance is initialized
    explicit operator bool() const { return static_cast<bool>(step_); }

  private:
    std::shared_ptr<const ParticleParams> particles_;
    std::shared_ptr<StepperInterface>     step_;
    std::vector<Primary>                  buffer_;

    EventId            event_id_;
    TrackId::size_type track_counter_{};

    size_type auto_flush_{};
    size_type max_steps_{};
};

//---------------------------------------------------------------------------//
} // namespace celeritas
