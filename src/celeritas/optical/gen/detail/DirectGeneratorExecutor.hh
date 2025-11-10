//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/DirectGeneratorExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/track/CoreStateCounters.hh"
#include "celeritas/track/Utils.hh"

#include "GeneratorAlgorithms.hh"
#include "../DirectGeneratorData.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    DirectGeneratorExecutor ...;
   \endcode
 */
struct DirectGeneratorExecutor
{
    CRefPtr<CoreParamsData, MemSpace::native> params;
    RefPtr<CoreStateData, MemSpace::native> state;
    NativeRef<DirectGeneratorStateData> const data;
    CoreStateCounters counters;

    inline CELER_FUNCTION void operator()(TrackSlotId tid) const;
    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        return (*this)(TrackSlotId{tid.unchecked_get()});
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 */
CELER_FUNCTION void DirectGeneratorExecutor::operator()(TrackSlotId tid) const
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);

    CoreTrackView vacancy(*params, *state, [&] {
        TrackSlotId idx{
            index_before(counters.num_vacancies, ThreadId(tid.get()))};
        return state->init.vacancies[idx];
    }());

    TrackInitializer const& init = data.initializers[ItemId<TrackInitializer>(
        index_before(counters.num_pending, ThreadId(tid.get())))];

    vacancy = init;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
