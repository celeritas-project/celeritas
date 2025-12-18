//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/LocalOpticalTrackOffload.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/optical/TrackInitializer.hh"
#include "celeritas/optical/Transporter.hh"
#include "accel/TrackOffloadInterface.hh"

#include "TrackOffloadInterface.hh"

class G4EventManager;

namespace celeritas
{
namespace optical
{
class CoreStateBase;
class Transporter;
}  // namespace optical

struct SetupOptions;
class SharedParams;

//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    LocalOpticalTrackOffload ...;
   \endcode
 */
class LocalOpticalTrackOffload final : public TrackOffloadInterface
{
  public:
    using TrackData = optical::TrackInitializer;
    // Construct in an invalid state
    LocalOpticalTrackOffload() = default;

    // Construct with shared (across threads) params
    LocalOpticalTrackOffload(SetupOptions const& options, SharedParams& params);

    //!@{
    //! \name Type aliases
    void Initialize(SetupOptions const&, SharedParams&) final;

    // Set the event ID and reseed the Celeritas RNG at the start of an event
    void InitializeEvent(int) final;

    // Transport all buffered tracks to completion
    void Flush() final;

    // Clear local data and return to an invalid state
    void Finalize() final;

    // Whether the class instance is initialized
    bool Initialized() const final { return static_cast<bool>(state_); }
    // Offload optical distribution data to Celeritas
    void Push(G4Track&) final;
    // Number of buffered tracks
    size_type GetBufferSize() const final { return pending_tracks_; }

    // Optical tracks pushed
    size_type num_pushed() const { return num_pushed_; }
    // Get accumulated action times
    MapStrDbl GetActionTime() const final;
    //!@}

  private:
    // Transport pending optical tracks
    std::shared_ptr<optical::Transporter> transport_;
    // Thread-local state data
    std::shared_ptr<optical::CoreStateBase> state_;

    std::vector<TrackData> buffer_;
    size_type pending_tracks_{};
    // Number of photons tracks to buffer before offloading
    size_type auto_flush_{};
    //  size_type num_pushed_{};
    // Diagnostics (thread-local)
    size_type num_pushed_{0};
    size_type num_flushed_{0};
    // size_type num_events_{0};
    //  Current event ID or manager for obtaining it
    UniqueEventId event_id_;
    G4EventManager* event_manager_{nullptr};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
