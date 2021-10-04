//----------------------------------*-hh-*-----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BetheHeitlerInteract.hh
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "celeritas_config.h"
#include "base/Assert.hh"
#include "../detail/BetheHeitler.hh"

namespace celeritas
{
namespace generated
{
void bethe_heitler_interact(
    const detail::BetheHeitlerHostRef&,
    const ModelInteractRefs<MemSpace::host>&);

void bethe_heitler_interact(
    const detail::BetheHeitlerDeviceRef&,
    const ModelInteractRefs<MemSpace::device>&);

#if !CELERITAS_USE_CUDA
inline void bethe_heitler_interact(
    const detail::BetheHeitlerDeviceRef&,
    const ModelInteractRefs<MemSpace::device>&)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

} // namespace generated
} // namespace celeritas
