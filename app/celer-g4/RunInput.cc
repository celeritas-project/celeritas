//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/RunInput.cc
//---------------------------------------------------------------------------//
#include "RunInput.hh"

#include "corecel/io/EnumStringMapper.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the physics list selection.
 */
char const* to_cstring(PhysicsListSelection value)
{
    static EnumStringMapper<PhysicsListSelection> const to_cstring_impl{
        "ftfp_bert",
        "celer_ftfp_bert",
        "celer_em",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the physics list selection.
 */
char const* to_cstring(SensitiveDetectorType value)
{
    static EnumStringMapper<SensitiveDetectorType> const to_cstring_impl{
        "none",
        "simple_calo",
        "event_hit",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Whether the run arguments are valid.
 */
RunInput::operator bool() const
{
    return !geometry_file.empty() && (primary_options || !event_file.empty())
           && physics_list < PhysicsListSelection::size_
           && (field == no_field() || field_options)
           && (num_track_slots > 0 && max_steps > 0 && initializer_capacity > 0
               && secondary_stack_factor > 0 && auto_flush > 0
               && auto_flush <= initializer_capacity)
           && (step_diagnostic_bins > 0 || !step_diagnostic);
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
