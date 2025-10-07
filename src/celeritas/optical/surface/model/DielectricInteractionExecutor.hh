//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/DielectricInteractionExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "DielectricInteractionData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * \todo This is a placeholder executor that does nothing until the dielectric
 * and UNIFIED calculators are implemented.
 */
struct DielectricInteractionExecutor
{
    NativeCRef<DielectricData> dielectric_data;
    NativeCRef<UnifiedReflectionData> reflection_data;

    inline CELER_FUNCTION void operator()(CoreTrackView&) const {}
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
