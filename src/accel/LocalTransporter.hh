//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/LocalTransporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Types.hh"
#include "corecel/io/Logger.hh"
#include "geocel/BoundingBox.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantTrackReconstruction.hh"

#include "TrackOffloadInterface.hh"

class G4EventManager;
class G4Track;

namespace celeritas
{
//---------------------------------------------------------------------------//
namespace detail
{
class HitProcessor;
}  // namespace detail

struct SetupOptions;
class CoreStateInterface;
class OffloadWriter;
class OpticalCollector;
class ParticleParams;
class SharedParams;
class StepperInterface;

//---------------------------------------------------------------------------//
/*!
 * Manage offloading of tracks to Celeritas.
 *
 * This class \em must be constructed locally on each worker
 * thread/task/stream, usually as a shared pointer that's accessible to:
 * - a run action (for initialization),
 * - an event action (to set the event ID and flush offloaded tracks at the end
 *   of the event)
 * - a tracking action (to try offloading every track)
 *
 * Stepper owns two fixed-capacity host primary buffers. LocalTransporter
 * validates and converts Geant4 tracks, pushes them into the producer buffer,
 * and keeps the corresponding Geant4 accounting. In device mode, a full
 * producer buffer is staged so Stepper can queue the H2D copy while Geant4
 * continues filling the other buffer. Transport later consumes the staged
 * buffer and, at event end, any remaining producer-buffer contents. Host mode
 * transports the producer buffer synchronously.
 *
 * \warning Due to Geant4 thread-local allocators, this class \em must be
 * finalized or destroyed on the same CPU thread in which is created and used!
 */
class LocalTransporter final : public TrackOffloadInterface
{
  public:
    // Construct in an invalid state
    LocalTransporter() = default;

    // Initialized with shared (across threads) params
    LocalTransporter(SetupOptions const& options, SharedParams& params);

    //!@{
    //! \name LocalOffload interface

    // Alternative to construction + move assignment
    inline void Initialize(SetupOptions const& options,
                           SharedParams& params) final;

    // Set the event ID and reseed the Celeritas RNG at the start of an event
    void InitializeEvent(int) final;

    // Transport all locally buffered and staged tracks to completion
    void Flush() final;

    // Clear local data and return to an invalid state
    void Finalize() final;

    // Whether the class instance is initialized
    bool Initialized() const final { return static_cast<bool>(step_); }

    // Number of locally buffered and staged tracks
    size_type GetBufferSize() const final;

    // Get accumulated action times
    MapStrDbl GetActionTime() const final;
    //!@}

    // Offload this track
    void Push(G4Track&) final;

    // Access core state data for user diagnostics
    CoreStateInterface const& GetState() const;

    // Access core state data for user diagnostics
    CoreStateInterface& GetState();

    //! Whether the class instance is initialized
    explicit operator bool() const { return this->Initialized(); }

  private:
    //// TYPES ////

    using SPOffloadWriter = std::shared_ptr<OffloadWriter>;
    using BBox = BoundingBox<double>;

    struct BufferAccum
    {
        double energy{0};  // MeV
        double lost_energy{0};  // MeV
        std::size_t lost_primaries{0};
    };

    struct RunAccum
    {
        std::size_t events{0};
        std::size_t flushes{0};
        std::size_t primaries{0};
        std::size_t steps{0};
        std::size_t lost_primaries{0};
        std::size_t hits{0};
    };

    //// HELPER FUNCTIONS ////

    void stage_buffered_primaries();
    void flush(bool include_buffered);

    //// DATA ////

    // Shared problem data and local configuration
    std::shared_ptr<ParticleParams const> particles_;
    BBox bbox_;
    SPOffloadWriter dump_primaries_;
    size_type max_step_iters_{};

    // Thread-local stepper data
    std::shared_ptr<StepperInterface> step_;
    BufferAccum buffered_accum_;
    BufferAccum staged_accum_;

    // Thread-local Geant4 integration data
    std::shared_ptr<detail::HitProcessor> hit_processor_;
    std::shared_ptr<GeantTrackReconstruction> track_reconstruction_;
    std::shared_ptr<OpticalCollector const> optical_;

    // Last seen event ID and manager for obtaining it
    int event_id_{-1};
    G4EventManager* event_manager_{nullptr};

    // Run summary diagnostics
    RunAccum run_accum_;
};

//---------------------------------------------------------------------------//
/*!
 * Helper for making initialization more obvious from user code.
 *
 * This gives it some symmetry with Finalize, which is provided as an
 * exception-friendly destructor.
 */
void LocalTransporter::Initialize(SetupOptions const& options,
                                  SharedParams& params)
{
    *this = LocalTransporter(options, params);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
