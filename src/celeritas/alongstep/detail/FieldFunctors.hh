//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/FieldFunctors.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Types.hh"
#include "celeritas/field/UniformFieldData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// CONDITIONS
//---------------------------------------------------------------------------//
/*!
 * Apply only to tracks in a volume with a field.
 */
struct IsInUniformField
{
    NativeCRef<UniformFieldParamsData> field;

    template<class T>
    CELER_FUNCTION bool operator()(T const& track) const
    {
        if (field.has_field.empty())
        {
            // Field is present in all volumes
            return true;
        }
        auto vol = track.geometry().impl_volume_id();
        CELER_ASSERT(vol < field.has_field.size());
        return field.has_field[vol];
    }
};

//---------------------------------------------------------------------------//
/*!
 * Apply to tracks in the uniform along-step action in volumes with field.
 */
struct IsAlongStepUniformField
{
    ActionId action;
    NativeCRef<UniformFieldParamsData> field;

    template<class T>
    CELER_FUNCTION bool operator()(T const& track) const
    {
        return IsAlongStepActionEqual{action}(track)
               && IsInUniformField{field}(track);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Apply to tracks in the uniform along-step action in volumes without field.
 */
struct IsAlongStepLinear
{
    ActionId action;
    NativeCRef<UniformFieldParamsData> field;

    template<class T>
    CELER_FUNCTION bool operator()(T const& track) const
    {
        return IsAlongStepActionEqual{action}(track)
               && !IsInUniformField{field}(track);
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
