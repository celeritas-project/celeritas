//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
    bool sync{false};  //!< Whether to synchronize device between actions

    // Loop control
    size_type max_steps{};
    bool store_track_counts{};  //!< Store track counts at each step

    StreamId stream_id{0};

    //! True if all params are assigned
    explicit operator bool() const
    {
        return params && num_track_slots > 0 && max_steps > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Tallied result and timing from transporting a single event.
 */
struct TransporterResult
{
    using VecReal = std::vector<real_type>;
    using VecCount = std::vector<size_type>;

    VecCount initializers;  //!< Num starting track initializers
    VecCount active;  //!< Num tracks active at beginning of step
    VecCount alive;  //!< Num living tracks at end of step
    VecReal step_times;  //!< Real time per step
    size_type num_track_slots{};  //!< Number of total track slots
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
    //!@}

  public:
    virtual ~TransporterBase() = 0;

    // Run a single step with no active states to "warm up"
    virtual void operator()() = 0;

    // Transport the input primaries and all secondaries produced
    virtual TransporterResult operator()(SpanConstPrimary primaries) = 0;
};

//---------------------------------------------------------------------------//
/*!
 * Transport a set of primaries to completion.
 */
template<MemSpace M>
class Transporter final : public TransporterBase
{
  public:
    //!@{
    //! \name Type aliases
    using MapStrReal = std::unordered_map<std::string, real_type>;
    //!@}

  public:
    // Construct from parameters
    explicit Transporter(TransporterInput inp);

    // Run a single step with no active states to "warm up"
    void operator()() final;

    // Transport the input primaries and all secondaries produced
    TransporterResult operator()(SpanConstPrimary primaries) final;

    // Get the accumulated action times
    MapStrReal get_action_times() const;

  private:
    std::shared_ptr<Stepper<M>> stepper_;
    size_type max_steps_;
    size_type num_streams_;
    bool store_track_counts_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
