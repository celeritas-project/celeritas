//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/generated/LivermorePEInteract.hh
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/em/data/LivermorePEData.hh" // IWYU pragma: associated

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
void livermore_pe_interact(
    celeritas::CoreParams const&,
    celeritas::CoreState<MemSpace::host>&,
    celeritas::LivermorePEHostRef const&);

void livermore_pe_interact(
    celeritas::CoreParams const&,
    celeritas::CoreState<MemSpace::device>&,
    celeritas::LivermorePEDeviceRef const&,
    celeritas::ActionId);

#if !CELER_USE_DEVICE
inline void livermore_pe_interact(
    celeritas::CoreParams const&,
    celeritas::CoreState<MemSpace::device>&,
    celeritas::LivermorePEDeviceRef const&,
    celeritas::ActionId)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

}  // namespace generated
}  // namespace celeritas
