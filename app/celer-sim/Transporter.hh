//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/Transporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
struct Primary;
template<MemSpace M>
class Stepper;
class CoreParams;
}  // namespace celeritas

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
//! Input parameters to the transporter.
struct TransporterInput
{
    // Stepper input
    std::shared_ptr<CoreParams const> params;
    size_type num_track_slots{};  //!< AKA max_num_tracks
    bool action_times{false};  //!< Whether to synchronize device between
                               //!< actions for timing

    // Loop control
    size_type max_steps{};
    bool store_track_counts{};  //!< Store track counts at each step
    bool store_step_times{};  //!< Store time elapsed for each step
    size_type log_progress{};  //!< CELER_LOG progress every N events

    StreamId stream_id{0};

    //! True if all params are assigned
    explicit operator bool() const
    {
        return params && num_track_slots > 0 && max_steps > 0
               && log_progress > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Tallied result and timing from transporting a single event.
 */
struct TransporterResult
{
    using VecCount = std::vector<size_type>;

    // Per-step diagnostics
    VecCount generated;  //!< Num primaries generated
    VecCount initializers;  //!< Num starting track initializers
    VecCount active;  //!< Num tracks active at beginning of step
    VecCount alive;  //!< Num living tracks at end of step
    std::vector<double> step_times;  //!< Real time per step

    // Always-on basic diagnostics
    size_type num_track_slots{};  //!< Number of total track slots
    size_type num_step_iterations{};  //!< Total number of step iterations
    size_type num_steps{};  //!< Total number of steps
    size_type num_tracks{};  //!< Total number of tracks
    size_type num_aborted{};  //!< Number of unconverged tracks
    size_type max_queued{};  //!< Maximum track initializer count
};

//---------------------------------------------------------------------------//
/*!
 * Interface class for transporting a set of primaries to completion.
 *
 * We might want to change this so that the transport result gets accumulated
 * over multiple calls rather than combining for a single operation, so
 * diagnostics would be an accessor and the "call" operator would be renamed
 * "transport". Such a change would imply making the diagnostics part of the
 * input parameters, which (for simplicity) isn't done yet.
 *
 * NOTE: there should be one transporter per "thread" state using shared
 * params.
 */
class TransporterBase
{
  public:
    //!@{
    //! \name Type aliases
    using SpanConstPrimary = Span<Primary const>;
    using MapStrDouble = std::unordered_map<std::string, double>;
    //!@}

  public:
    virtual ~TransporterBase() = 0;

    // Run a single step with no active states to "warm up"
    virtual void operator()() = 0;

    //! Transport the input primaries and all secondaries produced
    virtual TransporterResult operator()(SpanConstPrimary primaries) = 0;

    //! Accumulate action times into the map
    virtual void accum_action_times(MapStrDouble*) const = 0;
};

//---------------------------------------------------------------------------//
/*!
 * Transport a set of primaries to completion.
 */
template<MemSpace M>
class Transporter final : public TransporterBase
{
  public:
    // Construct from parameters
    explicit Transporter(TransporterInput inp);

    // Run a single step with no active states to "warm up"
    void operator()() final;

    // Transport the input primaries and all secondaries produced
    TransporterResult operator()(SpanConstPrimary primaries) final;

    // Accumulate action times into the map
    void accum_action_times(MapStrDouble*) const final;

  private:
    std::shared_ptr<Stepper<M>> stepper_;
    size_type max_steps_;
    size_type num_streams_;
    size_type log_progress_;
    bool store_track_counts_;
    bool store_step_times_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
