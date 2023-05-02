//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/generated/KleinNishinaInteract.hh
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/em/data/KleinNishinaData.hh" // IWYU pragma: associated

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
void klein_nishina_interact(
    celeritas::KleinNishinaHostRef const&,
    celeritas::CoreParams const&,
    celeritas::CoreState<MemSpace::host>&);

void klein_nishina_interact(
    celeritas::KleinNishinaDeviceRef const&,
    celeritas::CoreParams const&,
    celeritas::CoreState<MemSpace::device>&);

#if !CELER_USE_DEVICE
inline void klein_nishina_interact(
    celeritas::KleinNishinaDeviceRef const&,
    celeritas::CoreParams const&,
    celeritas::CoreState<MemSpace::device>&)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

}  // namespace generated
}  // namespace celeritas
