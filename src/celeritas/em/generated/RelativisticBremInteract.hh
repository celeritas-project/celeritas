//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/generated/RelativisticBremInteract.hh
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/em/data/RelativisticBremData.hh"

namespace celeritas
{
namespace generated
{
void relativistic_brem_interact(
    const celeritas::detail::RelativisticBremHostRef&,
    const CoreRef<MemSpace::host>&);

void relativistic_brem_interact(
    const celeritas::detail::RelativisticBremDeviceRef&,
    const CoreRef<MemSpace::device>&);

#if !CELER_USE_DEVICE
inline void relativistic_brem_interact(
    const celeritas::detail::RelativisticBremDeviceRef&,
    const CoreRef<MemSpace::device>&)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

} // namespace generated
} // namespace celeritas
