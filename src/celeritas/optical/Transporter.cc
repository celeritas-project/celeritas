//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Transporter.cc
//---------------------------------------------------------------------------//
#include "Transporter.hh"

#include <utility>

#include "corecel/cont/Range.hh"
#include "corecel/data/Ref.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stopwatch.hh"
#include "celeritas/phys/GeneratorRegistry.hh"

#include "CoreParams.hh"
#include "CoreState.hh"
#include "SimParams.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with problem parameters.
 */
Transporter::Transporter(Input&& inp) : data_(std::move(inp))
{
    CELER_EXPECT(data_.params);

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

    size_type num_step_iters{0};
    size_type num_steps{0};

    auto counters = state.sync_get_counters();

    // Store a pointer to aux data for timing results
    std::vector<double>* accum_time = nullptr;
    if (data_.action_times)
    {
        accum_time = &data_.action_times->state(*state.aux()).accum_time;
    }

    // Loop while photons are yet to be tracked
    while (counters.num_pending > 0 || counters.num_alive > 0)
    {
        ScopedProfiling profile_this{"optical-step"};
        // Loop through actions
        for (auto const& action : actions_->step())
        {
            ScopedProfiling profile_this{action->label()};
            Stopwatch get_time;
            action->step(*this->params(), state);
            if (accum_time)
            {
                if (M == MemSpace::device)
                {
                    device().stream(state.stream_id()).sync();
                }
                (*accum_time)[action->action_id().get()] += get_time();
            }
        }

        // No longer have a reference to the counters, so need to retrieve the
        // updated values
        counters = state.sync_get_counters();
        num_steps += counters.num_active;
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
                << counters.num_vacancies << " vacancies, and "
                << counters.num_pending << " queued";

            this->params()->gen_reg()->reset(*state.aux());
            state.reset();
            break;
        }
    }

    // Update statistics
    state.accum().steps += num_steps;
    state.accum().step_iters += num_step_iters;
    ++state.accum().flushes;
}

//---------------------------------------------------------------------------//
/*!
 * Get the accumulated action times.
 */
auto Transporter::get_action_times(AuxStateVec const& aux) const -> MapStrDbl
{
    if (data_.action_times)
    {
        return data_.action_times->get_action_times(aux);
    }
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
