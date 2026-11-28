//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/StepProcessor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/AsyncSnapshot.hh"
#include "corecel/sys/DeviceEvent.hh"
#include "celeritas/Types.hh"

#include "../GeantTrackReconstruction.hh"

class G4ParticleDefinition;
class G4Step;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Base class for transferring Celeritas step-like data to Geant4.
 *
 * This is a CRTP base that holds everything common to \c HitProcessor and
 * \c OnloadProcessor : the shared \c G4Step , the track reconstruction, and
 * the asynchronous device-to-host pipeline. The template parameters are:
 * - \c Derived : the concrete processor. It must provide
 *   \code
 *   static constexpr char const* process_label;  // profiling label
 *   void operator()(OutputT const&, size_type) const;  // process one step
 *   \endcode
 * - \c OutputT : host-side output container (e.g. \c DetectorStepOutput ). It
 *   must be contextually convertible to \c bool (true if nonempty) and have
 *   \c size() .
 * - \c HostStateT / \c DeviceStateT : host/device reference to the state data
 *   (e.g. \c HostRef<StepStateData> ). The free functions \c copy_steps ,
 *   \c reserve_compacted_steps , \c compact_steps_async and
 *   \c copy_compacted_steps_async must be findable by argument-dependent
 *   lookup for these types.
 *
 * \warning Derived classes must be thread-local and destroyed on the thread
 * that created them (Geant4 thread-local allocators).
 *
 * Host step data is copied and processed immediately by the call operator.
 * Device step data goes through three phases so that the device-to-host
 * transfer and the Geant4 processing can overlap with the next step:
 * 1. \b Compacted: the call operator enqueues the selection and compaction of
 *    steps into the state's scratch space, along with an asynchronous copy of
 *    the number of selected steps to pinned host memory, without
 *    synchronizing. Because the enqueued kernels read the gathered step data,
 *    it may be overwritten only by work enqueued later on the same stream.
 * 2. \b Copied: once the producing step is complete, \c copy_pending_steps
 *    reads the number of selected steps and enqueues the copy of the
 *    compacted data to pinned host storage on the device's \c memcpy_stream ,
 *    without synchronizing. The next compaction makes the state's stream wait
 *    for this copy, so the next step can be launched immediately.
 * 3. \b Processed: \c process_copied_steps waits for the copy, or
 *    \c process_copied_steps_if_ready polls it, and then processes the steps.
 *
 * These phases move each step through two buffers, the device \em scratch
 * space and the pinned host \em output, which act as a two-stage pipeline:
 * step \em n+1 can be compacted in the scratch while step \em n is still
 * copied in the output, waiting to be processed.
 * \verbatim
                      operator()(DeviceRef)
   scratch:   free ------------------------> compacted
               ^                                 |
               +------- copy_pending_steps ------+
                                                 | (enqueues the copy)
                                                 v
   output:    empty <----------------------- copied
                 process_copied_steps[_if_ready]
   \endverbatim
 *
 * When the scratch is "free", the device may still be reading it for the
 * previous copy: the next compaction waits for that copy on the device. The
 * calls allowed in each combined (scratch, output) state are:
 * - (free, empty): \c reserve , \c operator() and \c exchange_hits ;
 * - (compacted, empty): \c copy_pending_steps ;
 * - (free, copied): \c operator() and \c process_copied_steps[_if_ready] ;
 * - (compacted, copied): \c copy_pending_steps , which processes the output
 *   first, and \c process_copied_steps[_if_ready] .
 *
 * Copy and process calls in other states are no-ops, and
 * \c has_pending_steps is true in every state but (free, empty). The output
 * state is stored, while the scratch state is derived from whether compacted
 * step data is pending, so each buffer's state has a single source.
 *
 * \warning The next compaction waits only for this processor's copy. Since
 * all callbacks of a \c StepCollector share the scratch space, no other
 * callback may compact into it with \c compact_steps_async while a copy is in
 * flight, or the copied data would be silently replaced with later step data.
 */
template<class Derived, class OutputT, class HostStateT, class DeviceStateT>
class StepProcessor
{
  public:
    //!@{
    //! \name Type aliases
    using StepStateHostRef = HostStateT;
    using StepStateDeviceRef = DeviceStateT;
    using StepOutput = OutputT;
    using VecParticle = std::vector<G4ParticleDefinition const*>;
    //!@}

  public:
    // Wait for any pending device compaction before releasing its buffers
    ~StepProcessor();
    CELER_DEFAULT_MOVE_DELETE_COPY(StepProcessor);

    // Process CPU-generated steps
    void operator()(StepStateHostRef const&);

    // Allocate pinned host output for the largest possible device step
    void reserve(StepStateDeviceRef const&);

    // Enqueue compaction of device-generated steps for later processing
    void operator()(StepStateDeviceRef const&);

    // Enqueue the copy of compacted device steps after their step completes
    void copy_pending_steps();

    // Wait for and process copied device steps
    void process_copied_steps();

    // Process copied device steps if the copy has completed
    void process_copied_steps_if_ready();

    // Generate and call steps from an output (for testing)
    void operator()(StepOutput const& out) const;

    //! Whether device-generated step data is pending
    bool has_pending_steps() const noexcept
    {
        return this->scratch_state() != ScratchState::free
               || output_ != OutputState::empty;
    }

    // Get and reset the steps counted (generally once per event)
    inline size_type exchange_hits();

    //! Access local Geant4 track metadata reconstruction
    std::shared_ptr<GeantTrackReconstruction> const& track_reconstruction() const
    {
        return track_reconstruction_;
    }

  protected:
    // Construct the shared G4Step and (if particles are given) reconstruction
    explicit StepProcessor(VecParticle const& particles);

    //! Temporary CPU step information
    StepOutput steps_;
    //! Shared step object
    std::shared_ptr<G4Step> step_;
    //! Track reconstruction for step processing
    std::shared_ptr<GeantTrackReconstruction> track_reconstruction_;

  private:
    //! State of the device scratch space (see the class documentation)
    enum class ScratchState
    {
        free,  //!< Available for the next compaction
        compacted,  //!< Holds compacted steps awaiting a copy
    };

    //! State of the pinned host output (see the class documentation)
    enum class OutputState
    {
        empty,  //!< Available for the next copy
        copied,  //!< Holds copied steps awaiting processing
    };

    //! Device step data compacted in the scratch space, awaiting a copy
    StepStateDeviceRef pending_device_steps_;
    //! Number of compacted device steps, whose load completes the compaction
    AsyncSnapshot<size_type> num_selected_;
    //! Completion of the copy of compacted steps to \c steps_
    DeviceEvent copied_{nullptr};
    //! Whether \c steps_ holds copied device steps awaiting processing
    OutputState output_{OutputState::empty};
    //! Whether \c steps_ is reserved for the largest possible device step
    bool steps_reserved_{false};
    //! Accumulated number of steps
    size_type num_hits_{0};

    // Process steps already copied into temporary CPU storage
    void process_local_steps();

    //! State of the scratch space, derived from the pending step data
    ScratchState scratch_state() const noexcept
    {
        return pending_device_steps_ ? ScratchState::compacted
                                     : ScratchState::free;
    }

    //! Downcast to the concrete processor
    Derived& derived() { return static_cast<Derived&>(*this); }
    Derived const& derived() const
    {
        return static_cast<Derived const&>(*this);
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get and reset number of steps counted (generally once per event).
 */
template<class D, class O, class H, class S>
size_type StepProcessor<D, O, H, S>::exchange_hits()
{
    using namespace celeritas::literals;
    CELER_EXPECT(!this->has_pending_steps());
    return std::exchange(num_hits_, 0_sz);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
