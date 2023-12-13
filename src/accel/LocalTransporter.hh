//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/LocalTransporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/InitializedValue.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/phys/Primary.hh"

class G4Track;

namespace celeritas
{
namespace detail
{
class HitManager;
class OffloadWriter;
}  // namespace detail

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
    //!@{
    //! \name Type aliases
    using MapStrReal = std::unordered_map<std::string, real_type>;
    //!@}

  public:
    // Construct in an invalid state
    LocalTransporter() = default;

    // Initialized with shared (across threads) params
    LocalTransporter(SetupOptions const& options, SharedParams const& params);

    // Alternative to construction + move assignment
    inline void
    Initialize(SetupOptions const& options, SharedParams const& params);

    // Set the event ID
    void SetEventId(int);

    // Set the event ID and reseed the Celeritas RNG at the start of an event
    void InitializeEvent(int);

    // Offload this track
    void Push(G4Track const&);

    // Transport all buffered tracks to completion
    void Flush();

    // Clear local data and return to an invalid state
    void Finalize();

    // Get accumulated action times
    MapStrReal GetActionTime() const;

    // Number of buffered tracks
    size_type GetBufferSize() const { return buffer_.size(); }

    //! Whether the class instance is initialized
    explicit operator bool() const { return static_cast<bool>(step_); }

  private:
    using SPHitManger = std::shared_ptr<detail::HitManager>;
    using SPOffloadWriter = std::shared_ptr<detail::OffloadWriter>;

    struct HMFinalizer
    {
        StreamId sid;
        void operator()(SPHitManger& hm) const;
    };

    std::shared_ptr<ParticleParams const> particles_;
    std::shared_ptr<StepperInterface> step_;
    std::vector<Primary> buffer_;

    EventId event_id_;
    TrackId::size_type track_counter_{};

    size_type auto_flush_{};
    size_type max_steps_{};

    // Shared across threads to write flushed particles
    SPOffloadWriter dump_primaries_;
    // Shared pointer across threads, "finalize" called when clearing
    InitializedValue<SPHitManger, HMFinalizer> hit_manager_;
};

//---------------------------------------------------------------------------//
/*!
 * Helper for making initialization more obvious from user code.
 *
 * This gives it some symmetry with Finalize, which is provided as an
 * exception-friendly destructor.
 */
void LocalTransporter::Initialize(SetupOptions const& options,
                                  SharedParams const& params)
{
    *this = LocalTransporter(options, params);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
