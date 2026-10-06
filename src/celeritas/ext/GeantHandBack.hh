//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantHandBack.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Config.hh"

#include "celeritas/user/StepInterface.hh"

#include "detail/LocalProcessorSlots.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
namespace detail
{
class HandBackProcessor;
}
class GeantTrackReconstruction;

//---------------------------------------------------------------------------//
/*!
 * Reconstruct Geant4 tracks from Celeritas tracks handed back to Geant4.
 *
 * Any action can mark a track during a step with \c SimTrackView::hand_back ,
 * which kills it in Celeritas. This step interface gathers the end-of-step
 * state of those tracks only, and forwards it to a thread-local \c
 * detail::HandBackProcessor that reconstructs the corresponding \c G4Track
 * objects. It must be registered with its own \c StepCollector , separate from
 * any sensitive detector collector.
 *
 * Construction:
 * - Created alongside the step collector and shared across threads
 * - Thread-local processors are created with \c make_local_processor on the
 *   thread that uses them, sharing that thread's \c GeantTrackReconstruction
 *   (so that primary IDs map to the same Geant4 tracks as for hits)
 *
 * \sa detail::HandBackProcessor for the asynchronous processing contract.
 */
class GeantHandBack final : public StepInterface
{
  public:
    //!@{
    //! \name Type aliases
    using HandBackProcessor = detail::HandBackProcessor;
    using SPProcessor = std::shared_ptr<HandBackProcessor>;
    using SPTrackReconstruction = std::shared_ptr<GeantTrackReconstruction>;
    //!@}

  public:
    // Construct with the number of streams
    explicit GeantHandBack(StreamId::size_type num_streams);

    CELER_DELETE_COPY_MOVE(GeantHandBack);

    // Default destructor
    ~GeantHandBack();

    // Create local hand-back processor
    SPProcessor make_local_processor(StreamId sid, SPTrackReconstruction);

    // Select only handed-back tracks
    Filters filters() const final;

    // Selection of data required to reconstruct tracks
    StepSelection selection() const final;

    // Process CPU-generated tracks
    void process_steps(HostStepState) final;

    // Process device-generated tracks
    void process_steps(DeviceStepState) final;

  private:
    // Thread-local hand-back processors
    detail::LocalProcessorSlots<HandBackProcessor> processors_;
};

#if !CELERITAS_USE_GEANT4

inline GeantHandBack::GeantHandBack(StreamId::size_type)
{
    CELER_NOT_CONFIGURED("Geant4");
}

inline GeantHandBack::~GeantHandBack() = default;

inline GeantHandBack::SPProcessor GeantHandBack::make_local_processor(
    StreamId, SPTrackReconstruction)
{
    CELER_ASSERT_UNREACHABLE();
}

inline GeantHandBack::Filters GeantHandBack::filters() const
{
    CELER_ASSERT_UNREACHABLE();
}

inline StepSelection GeantHandBack::selection() const
{
    CELER_ASSERT_UNREACHABLE();
}

inline void GeantHandBack::process_steps(HostStepState)
{
    CELER_ASSERT_UNREACHABLE();
}

inline void GeantHandBack::process_steps(DeviceStepState)
{
    CELER_ASSERT_UNREACHABLE();
}

#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
