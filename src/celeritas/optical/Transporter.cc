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
#include "celeritas/phys/GeneratorRegistry.hh"

#include "CoreParams.hh"
#include "CoreState.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with problem parameters.
 */
Transporter::Transporter(Input&& inp)
    : params_(std::move(inp.params)), max_step_iters_(inp.max_step_iters)
{
    CELER_EXPECT(params_);

    actions_ = std::make_shared<ActionGroupsT>(*params_->action_reg());
}

//---------------------------------------------------------------------------//
/*!
 * Transport all pending optical tracks on the host.
 */
void Transporter::operator()(CoreStateHost& state) const
{
    this->transport_impl(state);
}

//---------------------------------------------------------------------------//
/*!
 * Transport all pending optical tracks on the device.
 */
void Transporter::operator()(CoreStateDevice& state) const
{
    this->transport_impl(state);
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

    auto const& counters = state.counters();

    // Loop while photons are yet to be tracked
    while (counters.num_pending > 0 || counters.num_alive > 0)
    {
        ScopedProfiling profile_this{"optical-step"};
        // Loop through actions
        for (auto const& action : actions_->step())
        {
            ScopedProfiling profile_this{action->label()};
            action->step(*params_, state);
        }

        num_steps += counters.num_active;
        if (CELER_UNLIKELY(++num_step_iters == max_step_iters_))
        {
            CELER_LOG_LOCAL(error)
                << "Exceeded step count of " << max_step_iters_
                << ": aborting optical transport loop with "
                << counters.num_generated << " generated tracks, "
                << counters.num_active << " active tracks, "
                << counters.num_alive << " alive tracks, "
                << counters.num_vacancies << " vacancies, and "
                << counters.num_pending << " queued";

            params_->gen_reg()->reset(*state.aux());
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
}  // namespace optical
}  // namespace celeritas
