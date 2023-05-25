//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/generated/RayleighInteract.hh
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/em/data/RayleighData.hh" // IWYU pragma: associated

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
void rayleigh_interact(
    celeritas::CoreParams const&,
    celeritas::CoreState<MemSpace::host>&,
    celeritas::RayleighHostRef const&);

void rayleigh_interact(
    celeritas::CoreParams const&,
    celeritas::CoreState<MemSpace::device>&,
    celeritas::RayleighDeviceRef const&,
    celeritas::ActionId);

#if !CELER_USE_DEVICE
inline void rayleigh_interact(
    celeritas::CoreParams const&,
    celeritas::CoreState<MemSpace::device>&,
    celeritas::RayleighDeviceRef const&,
    celeritas::ActionId)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

}  // namespace generated
}  // namespace celeritas
