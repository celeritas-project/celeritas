//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/LocalTransporter.hh
//! \sa TrackingManagerIntegration.test.cc
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
struct StepperResult;

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
 * \par Primary buffering
 *
 * Stepper owns two fixed-capacity host primary buffers and the device event
 * that protects the source of an asynchronous H2D copy. LocalTransporter
 * validates and converts Geant4 tracks, pushes accepted primaries into the
 * Stepper producer buffer, and keeps the corresponding Geant4 energy and loss
 * accounting. Primaries outside the Celeritas geometry are counted as lost but
 * are not added to either Stepper buffer.
 *
 * \par Device execution
 *
 * When the producer buffer first reaches capacity, \c Push stages it and
 * immediately calls \c StepperInterface::async. Same-stream ordering ensures
 * that the primary-copy completes before the stepping kernels use it. The
 * Stepper retains the staged host storage until the copy completes, while
 * Geant4 can continue filling the second buffer.
 *
 * At most one step, one staged successor, and one producer batch can be
 * pending. If the producer buffer fills while a step is in flight, \c Push
 * applies backpressure by consuming the previous result and advancing the
 * existing Celeritas tracks only until the initializer queue has room for the
 * full producer buffer. If a successor is already staged, it is launched first
 * so its former host buffer can become the next staging source. Active tail
 * tracks may remain when a new batch is staged and launched.
 * Calls to \c stage_primaries and \c async can still block on synchronization
 * internal to the current Stepper implementation.
 * This admission check reserves space for the incoming primaries; secondaries
 * generated during the next step remain subject to the existing initializer
 * capacity validation.
 * Before accepting each subsequent track, \c Push polls a pending step. If it
 * is ready, its result is consumed and another step is launched whenever
 * existing transport remains or another batch is staged. This allows device
 * transport to progress while Geant4 continues producing primaries. New
 * primaries are admitted only when the initializer queue also has room for the
 * maximum number of secondaries that can be produced by the next step.
 *
 * \par Event completion
 *
 * At event end, \c Flush first advances an in-flight step until any partially
 * filled producer buffer fits, then stages and launches that buffer before
 * synchronously stepping all transport to completion. Hit processing and
 * Geant4 track reconstruction are kept alive across the asynchronous work and
 * cleared only after this drain completes. A flush with only rejected
 * primaries still reports and clears their loss accounting.
 *
 * Host mode has the same buffering interface but no asynchronous overlap: a
 * full producer buffer calls \c Flush and is transported to completion before
 * \c Push returns.
 *
 * \internal
 *
 * LocalTransporter accounting follows the Stepper primary lifecycle:
 *
 * | State | Stepper state | Local accounting |
 * | ----- | ------------- | ---------------- |
 * | Producer | Primaries accepted by \c push_primary | \c buffered_accum_ |
 * | Staged | H2D copy queued, not submitted | \c staged_accum_ |
 * | In flight | Step result is valid | \c in_flight_accum_ |
 * | Complete | Step result consumed | Added to \c run_accum_ |
 *
 * \c GetBufferSize returns accepted primaries in the producer, staged, and
 * in-flight phases. It does not count rejected primaries, active Celeritas
 * tracks, or generated secondaries. The \c transport_active_ flag records
 * whether the last consumed result requires another step, while \c valid on
 * the Stepper records a currently pending result. The step iteration count
 * spans all overlapping primary batches and resets only after transport
 * becomes empty. Calling \c Finalize requires both states to be idle, normally
 * by first calling \c Flush.
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

    // Transport all queued and in-flight tracks to completion
    void Flush() final;

    // Clear local data and return to an invalid state
    void Finalize() final;

    // Whether the class instance is initialized
    bool Initialized() const final { return static_cast<bool>(step_); }

    // Number of buffered, staged, and in-flight primaries
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
        size_type primaries{0};
        double energy{0};  // MeV
        double lost_energy{0};  // MeV
        std::size_t lost_primaries{0};

        bool empty() const { return primaries == 0 && lost_primaries == 0; }
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
    size_type available_primary_capacity(size_type queued) const;
    void launch_step();
    StepperResult complete_step();
    void advance_if_ready();
    StepperResult advance_transport();
    void wait_for_initializer_capacity();
    void drain_transport();

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
    BufferAccum in_flight_accum_;
    size_type step_iters_{0};
    bool transport_active_{false};

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
