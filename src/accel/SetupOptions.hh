//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SetupOptions.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Control options for initializing Celeritas.
 */
struct SetupOptions
{
    using size_type = unsigned int;
    using real_type = double;

    //! Don't limit the number of steps
    static constexpr size_type no_max_steps()
    {
        return static_cast<size_type>(-1);
    }
    // TODO: names of sensitive detectors
    // TODO: along-step construction option/callback

    size_type    max_num_tracks{};
    size_type    max_steps = no_max_steps();
    size_type    initializer_capacity{};
    real_type    secondary_stack_factor{};
};

//---------------------------------------------------------------------------//
} // namespace celeritas
