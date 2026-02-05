//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/MockInteractAction.cc
//---------------------------------------------------------------------------//
#include "MockInteractAction.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "MockInteractExecutor.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Construct with number of secondaries and post-interact state.
 */
MockInteractAction::MockInteractAction(
    ActionId id,
    std::vector<size_type> const& num_secondaries,
    std::vector<bool> const& alive)
    : id_(id)
{
    CELER_EXPECT(id_);
    CELER_EXPECT(!num_secondaries.empty());
    CELER_EXPECT(num_secondaries.size() == alive.size());

    HostVal<MockInteractData> data;
    make_builder(&data.num_secondaries)
        .insert_back(num_secondaries.begin(), num_secondaries.end());
    make_builder(&data.alive).insert_back(alive.begin(), alive.end());
    CELER_ASSERT(data);

    data_ = ParamsDataStore<MockInteractData>{std::move(data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
void MockInteractAction::step(CoreParams const& params,
                              CoreStateHost& state) const
{
    auto execute = make_active_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        MockInteractExecutor{data_.ref<MemSpace::native>()});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
/*!
 * Get the number of secondaries.
 */
Span<size_type const> MockInteractAction::num_secondaries() const
{
    return data_.host_ref().num_secondaries[AllItems<size_type>{}];
}

//---------------------------------------------------------------------------//
/*!
 * Get true/false values for the pending track states.
 */
Span<char const> MockInteractAction::alive() const
{
    return data_.host_ref().alive[AllItems<char>{}];
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
