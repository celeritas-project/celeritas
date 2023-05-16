//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/Transporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Types.hh"

#include "Runner.hh"

namespace celeritas
{
struct Primary;
template<MemSpace M>
class Stepper;
class CoreParams;
}

namespace demo_loop
{
//---------------------------------------------------------------------------//
//! Input parameters to the transporter.
struct TransporterInput
{
    using size_type = celeritas::size_type;
    using CoreParams = celeritas::CoreParams;

    // Stepper input
    std::shared_ptr<CoreParams const> params;
    size_type num_track_slots{};  //!< AKA max_num_tracks
    bool sync{false};  //!< Whether to synchronize device between actions

    // Loop control
    size_type max_steps{};

    celeritas::StreamId stream_id{0};

    //! True if all params are assigned
    explicit operator bool() const
    {
        return params && num_track_slots > 0 && max_steps > 0;
    }
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
    using SpanConstPrimary = celeritas::Span<const celeritas::Primary>;
    using CoreParams = celeritas::CoreParams;
    using ActionId = celeritas::ActionId;
    using TransporterResult = RunnerResult;
    //!@}

  public:
    virtual ~TransporterBase() = 0;

    // Transport the input primaries and all secondaries produced
    virtual TransporterResult operator()(SpanConstPrimary primaries) = 0;
};

//---------------------------------------------------------------------------//
/*!
 * Transport a set of primaries to completion.
 */
template<celeritas::MemSpace M>
class Transporter final : public TransporterBase
{
  public:
    // Construct from parameters
    explicit Transporter(TransporterInput inp);

    // Transport the input primaries and all secondaries produced
    TransporterResult operator()(SpanConstPrimary primaries) final;

  private:
    std::shared_ptr<celeritas::Stepper<M>> stepper_;
    celeritas::size_type max_steps_;
    bool time_stream_;
};

//---------------------------------------------------------------------------//
}  // namespace demo_loop
