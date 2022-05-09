//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file physics/em/generated/MollerBhabhaInteract.hh
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "base/Assert.hh"
#include "base/Macros.hh"
#include "sim/CoreTrackData.hh"
#include "../detail/MollerBhabhaData.hh"

namespace celeritas
{
namespace generated
{
void moller_bhabha_interact(
    const celeritas::detail::MollerBhabhaHostRef&,
    const CoreRef<MemSpace::host>&);

void moller_bhabha_interact(
    const celeritas::detail::MollerBhabhaDeviceRef&,
    const CoreRef<MemSpace::device>&);

#if !CELER_USE_DEVICE
inline void moller_bhabha_interact(
    const celeritas::detail::MollerBhabhaDeviceRef&,
    const CoreRef<MemSpace::device>&)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

} // namespace generated
} // namespace celeritas
