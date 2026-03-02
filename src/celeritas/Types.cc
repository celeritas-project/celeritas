//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Types.cc
//---------------------------------------------------------------------------//
#include "Types.hh"

#include "corecel/io/EnumStringMapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a state of matter.
 */
char const* to_cstring(MatterState value)
{
    static EnumStringMapper<MatterState> const to_cstring_impl{
        "unspecified",
        "solid",
        "liquid",
        "gas",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a track status
 */
char const* to_cstring(TrackStatus value)
{
    static EnumStringMapper<TrackStatus> const to_cstring_impl{
        "inactive",
        "initializing",
        "alive",
        "errored",
        "killed",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a track ordering policy.
 */
char const* to_cstring(TrackOrder value)
{
    static EnumStringMapper<TrackOrder> const to_cstring_impl{
        "none",
        "init_charge",
        "reindex_shuffle",
        "reindex_status",
        "reindex_particle_type",
        "reindex_along_step_action",
        "reindex_step_limit_action",
        "reindex_both_action",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the MSC step limit algorithm.
 */
char const* to_cstring(MscStepLimitAlgorithm value)
{
    static EnumStringMapper<MscStepLimitAlgorithm> const to_cstring_impl{
        "minimal",
        "safety",
        "safety_plus",
        "distance_to_boundary",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the nuclear form factor model.
 */
char const* to_cstring(NuclearFormFactorType value)
{
    static EnumStringMapper<NuclearFormFactorType> const to_cstring_impl{
        "none",
        "flat",
        "exponential",
        "gaussian",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
