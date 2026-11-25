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
#include "celeritas/track/TrackFunctors.hh"

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
/*!
 * Whether the whole step of a track fits inside its safety sphere.
 *
 * The sphere is centered on the track's position at the start of the step
 * (see \c SafetyApplier ), so the check uses the center rather than the
 * current position. It therefore has the same answer after the track has been
 * propagated, which lets one kernel skip the tracks another has already moved.
 */
template<class T>
CELER_FUNCTION bool is_step_in_safety_sphere(T const& track,
                                             real_type tolerance)
{
    auto const& sphere = track.physics_step().safety_sphere();
    return sphere.contains(sphere.center,
                           track.sim().step_length() + tolerance);
}

//---------------------------------------------------------------------------//
/*!
 * Apply to tracks satisfying a condition whose step fits in the sphere.
 */
template<class C>
struct IsStepInSafetySphere
{
    C condition;
    real_type tolerance;

    template<class T>
    CELER_FUNCTION bool operator()(T const& track) const
    {
        return condition(track) && is_step_in_safety_sphere(track, tolerance);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Apply to tracks satisfying a condition that don't use the sphere.
 *
 * If the safety sphere is disabled, this applies to all tracks that satisfy
 * the condition.
 */
template<class C>
struct IsStepOutsideSafetySphere
{
    C condition;
    real_type tolerance;
    bool use_sphere;

    template<class T>
    CELER_FUNCTION bool operator()(T const& track) const
    {
        return condition(track)
               && !(use_sphere && is_step_in_safety_sphere(track, tolerance));
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
