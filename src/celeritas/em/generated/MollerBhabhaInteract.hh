//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/generated/MollerBhabhaInteract.hh
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/em/data/MollerBhabhaData.hh" // IWYU pragma: associated
#include "celeritas/global/CoreTrackData.hh"

namespace celeritas
{
namespace generated
{
void moller_bhabha_interact(
    celeritas::MollerBhabhaHostRef const&,
    celeritas::CoreRef<celeritas::MemSpace::host> const&);

void moller_bhabha_interact(
    celeritas::MollerBhabhaDeviceRef const&,
    celeritas::CoreRef<celeritas::MemSpace::device> const&);

#if !CELER_USE_DEVICE
inline void moller_bhabha_interact(
    celeritas::MollerBhabhaDeviceRef const&,
    celeritas::CoreRef<celeritas::MemSpace::device> const&)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

}  // namespace generated
}  // namespace celeritas
