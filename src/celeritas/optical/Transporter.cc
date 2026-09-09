//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Transporter.cc
//---------------------------------------------------------------------------//
#include "Transporter.hh"

#include <utility>

#include "corecel/io/Logger.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stopwatch.hh"
#include "celeritas/phys/GeneratorRegistry.hh"  // IWYU pragma: keep

#include "CoreParams.hh"
#include "CoreState.hh"
#include "SimParams.hh"  // IWYU pragma: keep

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with problem parameters.
 */
Transporter::Transporter(Input&& inp) : input_(std::move(inp))
{
    CELER_EXPECT(input_.params);

    actions_ = std::make_shared<ActionGroupsT>(*this->params()->action_reg());
}

//---------------------------------------------------------------------------//
/*!
 * Transport all pending optical tracks on the host.
 */
void Transporter::operator()(CoreStateBase& state) const
{
    if (auto* s = dynamic_cast<CoreStateHost*>(&state))
    {
        return this->transport_impl(*s);
    }
    else if (auto* s = dynamic_cast<CoreStateDevice*>(&state))
    {
        return this->transport_impl(*s);
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Transport all pending optical tracks.
 */
template<MemSpace M>
void Transporter::transport_impl(CoreState<M>& state) const
{
    CELER_EXPECT(state.aux());

    CELER_LOG_LOCAL(status) << "Transporting on " << to_cstring(M);

    size_type num_step_iters{0};
    size_type num_steps{0};

    auto counters = state.sync_get_counters();

    // Store a pointer to aux data for timing results
    std::vector<double>* accum_time = nullptr;
    if (input_.action_times)
    {
        accum_time = &input_.action_times->state(*state.aux()).accum_time;
    }

    // Loop while photons are yet to be tracked
    while (counters.num_pending > 0 || counters.num_alive > 0)
    {
        ScopedProfiling profile_this{"step"};
        Stopwatch get_step_time;

        // Loop through actions
        for (auto const& action : actions_->step())
        {
            ScopedProfiling profile_this{action->label()};
            Stopwatch get_action_time;
            action->step(*this->params(), state);
            if (accum_time)
            {
                if (M == MemSpace::device)
                {
                    device().stream(state.stream_id()).sync();
                }
                (*accum_time)[action->action_id().get()] += get_action_time();
            }
        }

        // No longer have a reference to the counters, so need to retrieve the
        // updated values
        counters = state.sync_get_counters();
        num_steps += counters.num_active;

        // Record the step time
        if (input_.step_times)
        {
            if (M == MemSpace::device)
            {
                device().stream(state.stream_id()).sync();
            }
            auto& step_times = input_.step_times->state(*state.aux()).time;
            step_times.push_back(get_step_time());
        }

        if (CELER_UNLIKELY(++num_step_iters
                           == this->params()->sim()->max_step_iters()))
        {
            CELER_LOG_LOCAL(error)
                << "Exceeded step count of "
                << this->params()->sim()->max_step_iters()
                << ": aborting optical transport loop with "
                << counters.num_generated << " generated tracks, "
                << counters.num_active << " active tracks, "
                << counters.num_alive << " alive tracks, "
                << counters.num_vacancies << " vacancies, "
                << counters.num_pending << " queued, " << counters.num_cut
                << " cut, and " << counters.num_errored << " errored";
            // Add all untransported tracks to 'cut' counter
            counters.num_cut += counters.num_active + counters.num_pending;

            this->params()->gen_reg()->reset(*state.aux());
            state.reset();
            break;
        }
    }
    if (counters.num_cut > 0)
    {
        CELER_LOG_LOCAL(warning) << "Terminated " << counters.num_cut
                                 << " optical tracks due to cutoff/limits";
    }
    if (counters.num_errored > 0)
    {
        CELER_LOG_LOCAL(error) << "Terminated " << counters.num_errored
                               << " optical tracks that errored";
    }

    // Update statistics
    state.accum().steps += num_steps;
    state.accum().step_iters += num_step_iters;
    ++state.accum().flushes;

    // Accumulate cut/error counters from the last synchronized counters
    state.accum().num_cut += counters.num_cut;
    state.accum().num_errored += counters.num_errored;
}

//---------------------------------------------------------------------------//
/*!
 * Get the accumulated action times.
 */
auto Transporter::get_action_times(AuxStateVec const& aux) const -> MapStrDbl
{
    if (input_.action_times)
    {
        return input_.action_times->get_action_times(aux);
    }
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Get the recorded step times.
 */
auto Transporter::get_step_times(AuxStateVec const& aux) const -> VecDbl
{
    if (input_.step_times)
    {
        return input_.step_times->state(aux).time;
    }
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
