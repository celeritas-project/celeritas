//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/generated/CombinedBremInteract.hh
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/em/data/CombinedBremData.hh" // IWYU pragma: associated

namespace celeritas
{
class CoreParams;
template<MemSpace M>
class CoreState;
}

namespace celeritas
{
namespace generated
{
void combined_brem_interact(
    celeritas::CombinedBremHostRef const&,
    celeritas::CoreParams const&,
    celeritas::CoreState<MemSpace::host>&);

void combined_brem_interact(
    celeritas::CombinedBremDeviceRef const&,
    celeritas::CoreParams const&,
    celeritas::CoreState<MemSpace::device>&);

#if !CELER_USE_DEVICE
inline void combined_brem_interact(
    celeritas::CombinedBremDeviceRef const&,
    celeritas::CoreParams const&,
    celeritas::CoreState<MemSpace::device>&)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

}  // namespace generated
}  // namespace celeritas
