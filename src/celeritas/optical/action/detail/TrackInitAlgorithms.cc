//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/TrackInitAlgorithms.cc
//---------------------------------------------------------------------------//
#include "TrackInitAlgorithms.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Compact the \c TrackSlotIds of the inactive tracks.
 *
 * \return Number of vacant track slots
 */
size_type copy_if_vacant(TrackStatusRef<MemSpace::host> const& status,
                         TrackSlotRef<MemSpace::host> const& vacancies,
                         StreamId)
{
    CELER_EXPECT(status.size() == vacancies.size());

    auto* data = status.data().get();
    auto* result = vacancies.data().get();

    size_type tid = 0;
    auto* const stop = data + status.size();
    for (; data != stop; ++data)
    {
        if (IsVacant{}(*data))
        {
            *result++ = TrackSlotId{tid};
        }
        ++tid;
    }
    return result - vacancies.data().get();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
