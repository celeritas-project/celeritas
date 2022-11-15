//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/generated/ProcessPrimaries.cc
//! \note Auto-generated by gen-trackinit.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "celeritas/track/detail/ProcessPrimariesLauncher.hh" // IWYU pragma: associated
#include "corecel/sys/ThreadId.hh"
#include "corecel/Types.hh"

namespace celeritas
{
namespace generated
{
void process_primaries(
    const CoreHostRef& core_data,
    const Span<const Primary> primaries)
{
    detail::ProcessPrimariesLauncher<MemSpace::host> launch(core_data, primaries);
    #pragma omp parallel for
    for (ThreadId::size_type i = 0; i < primaries.size(); ++i)
    {
        launch(ThreadId{i});
    }
}

} // namespace generated
} // namespace celeritas
