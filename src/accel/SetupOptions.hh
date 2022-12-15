//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SetupOptions.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

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

    // TODO: geometry should be exported directly from Geant4 (or written to
    // temporary GDML and re-read)
    std::string geometry_file;
    std::string hepmc3_file;

    size_type max_num_tracks{};
    size_type max_num_events{};
    size_type max_steps = no_max_steps();
    size_type initializer_capacity{};
    real_type secondary_stack_factor{};
    bool      sync{};
};

//---------------------------------------------------------------------------//
} // namespace celeritas
