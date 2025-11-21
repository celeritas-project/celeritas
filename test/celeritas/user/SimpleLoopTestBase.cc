//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SimpleLoopTestBase.cc
//---------------------------------------------------------------------------//
#include "SimpleLoopTestBase.hh"

#include "corecel/Types.hh"
#include "corecel/io/LogContextException.hh"
#include "celeritas/global/Stepper.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Number primaries given number track slots.
 *
 * Default is passthrough.
 */
size_type SimpleLoopTestBase::initial_occupancy(size_type num_tracks) const
{
    return num_tracks;
}

//---------------------------------------------------------------------------//
/*!
 * Run a stepping loop with the core data.
 */
template<MemSpace M>
void SimpleLoopTestBase::run_impl(size_type num_tracks, size_type num_steps)
{
    StepperInput step_inp;
    step_inp.params = this->core();
    step_inp.stream_id = StreamId{0};
    step_inp.num_track_slots = num_tracks;
    step_inp.actions = std::make_shared<ActionSequence>(
        *this->action_reg(), ActionSequence::Options{});

    if constexpr (M == MemSpace::device)
    {
        device().create_streams(1);
    }

    Stepper<M> step(step_inp);
    LogContextException log_context{this->output_reg().get()};

    size_type num_primaries = this->initial_occupancy(num_tracks);
    // Initial step
    auto primaries = this->make_primaries(num_primaries);
    StepperResult count;
    CELER_TRY_HANDLE(count = step(make_span(primaries)), log_context);

    while (count && --num_steps > 0)
    {
        CELER_TRY_HANDLE(count = step(), log_context);
    }
}

template void
    SimpleLoopTestBase::run_impl<MemSpace::host>(size_type, size_type);
template void
    SimpleLoopTestBase::run_impl<MemSpace::device>(size_type, size_type);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
