//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/MockInteractAction.cc
//---------------------------------------------------------------------------//
#include "MockInteractAction.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/KernelContextException.hh"
#include "celeritas/global/TrackLauncher.hh"

#include "MockInteractImpl.hh"

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

    data_ = CollectionMirror<MockInteractData>{std::move(data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
void MockInteractAction::execute(ParamsHostCRef const& params,
                                 StateHostRef& state) const
{
    CELER_EXPECT(params && state);
    CELER_EXPECT(state.size() == data_.host_ref().size());

    MultiExceptionHandler capture_exception;
    auto launch = make_active_track_launcher(
        params, state, apply_mock_interact, data_.host_ref());
#ifdef _OPENMP
#    pragma omp parallel for
#endif
    for (size_type i = 0; i < state.size(); ++i)
    {
        CELER_TRY_HANDLE_CONTEXT(
            launch(ThreadId{i}),
            capture_exception,
            KernelContextException(params, state, ThreadId{i}, this->label()));
    }
    log_and_rethrow(std::move(capture_exception));
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
