//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "corecel/io/Logger.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/phys/Primary.hh"

class G4Track;

namespace celeritas
{
//---------------------------------------------------------------------------//
namespace detail
{
class HitProcessor;
class OffloadWriter;
}  // namespace detail

struct SetupOptions;
class SharedParams;

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
 * \warning Due to Geant4 thread-local allocators, this class \em must be
 * finalized or destroyed on the same CPU thread in which is created and used!
 *
 * \todo Rename \c LocalOffload or something?
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
    LocalTransporter(SetupOptions const& options, SharedParams& params);

    // Alternative to construction + move assignment
    inline void Initialize(SetupOptions const& options, SharedParams& params);

    // Set the event ID and reseed the Celeritas RNG (remove in v0.6)
    [[deprecated]] void SetEventId(int id) { this->InitializeEvent(id); }

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
    using SPOffloadWriter = std::shared_ptr<detail::OffloadWriter>;

    std::shared_ptr<ParticleParams const> particles_;
    std::shared_ptr<StepperInterface> step_;
    std::vector<Primary> buffer_;
    std::shared_ptr<detail::HitProcessor> hit_processor_;

    UniqueEventId event_id_;

    size_type auto_flush_{};
    size_type max_step_iters_{};
    double buffer_energy_{0};

    std::size_t accum_num_events_{0};
    std::size_t accum_num_primaries_{0};
    std::size_t accum_num_steps_{0};

    // Shared across threads to write flushed particles
    SPOffloadWriter dump_primaries_;
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
