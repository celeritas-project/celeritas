//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/UniformFieldPropagatorFactory.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/field/DormandPrinceStepper.hh"  // IWYU pragma: associated
#include "celeritas/field/MakeMagFieldPropagator.hh"  // IWYU pragma: associated
#include "celeritas/field/UniformField.hh"  // IWYU pragma: associated
#include "celeritas/field/UniformFieldData.hh"  // IWYU pragma: associated

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Propagate a track in a uniform magnetic field.
 */
struct UniformFieldPropagatorFactory
{
#if CELER_USE_DEVICE
    inline static constexpr int max_block_size = 256;
#endif

    CELER_FUNCTION decltype(auto) operator()(CoreTrackView const& track) const
    {
        return make_mag_field_propagator<DormandPrinceStepper>(
            UniformField(field.field),
            field.options,
            track.particle(),
            track.geometry());
    }

    static CELER_CONSTEXPR_FUNCTION bool tracks_can_loop() { return true; }

    //// DATA ////

    NativeCRef<UniformFieldParamsData> field;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
