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
#include "celeritas/optical/gen/DirectGeneratorAction.hh"

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
 * Offload Geant4 optical photon tracks to Celeritas.
 */
class LocalOpticalTrackOffload final : public TrackOffloadInterface
{
  public:
    //!@{
    //! \name Type aliases
    using TrackData = optical::TrackInitializer;
    //!@}

  public:
    // Construct in an invalid state
    LocalOpticalTrackOffload() = default;

    // Construct with shared (across threads) params
    LocalOpticalTrackOffload(SetupOptions const& options, SharedParams& params);

    //!@{
    //! \name TrackOffloadInterface

    // Initialize with options and shared data
    void Initialize(SetupOptions const&, SharedParams&) final;

    // Set the event ID and reseed the Celeritas RNG at the start of an event
    void InitializeEvent(int) final;

    // Transport all buffered tracks to completion
    void Flush() final;

    // Clear local data and return to an invalid state
    void Finalize() final;

    // Whether the class instance is initialized
    bool Initialized() const final { return static_cast<bool>(transport_); }

    // Number of buffered tracks
    size_type GetBufferSize() const final { return buffer_.size(); }

    // Get accumulated action times
    MapStrDbl GetActionTime() const final;
    //!@}

    // Offload optical photon track to Celeritas
    void Push(G4Track&) final;

    // Number of optical tracks pushed to offload
    size_type num_pushed() const { return num_pushed_; }

  private:
    // Transport pending optical tracks
    std::shared_ptr<optical::Transporter> transport_;

    // Thread-local state data
    std::shared_ptr<optical::CoreStateBase> state_;

    // Generator action for injecting buffered tracks into optical state
    std::shared_ptr<optical::DirectGeneratorAction const> direct_gen_;

    // Buffered tracks for offloading
    std::vector<TrackData> buffer_;

    // Number of photons tracks to buffer before offloading
    size_type auto_flush_{};

    // Accumulated number of optical photon tracks
    size_type num_pushed_{};

    // Accumulated number of tracks pushed over flushes
    size_type num_flushed_{};

    //  Current event ID or manager for obtaining it
    UniqueEventId event_id_;
    G4EventManager* event_manager_{nullptr};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
