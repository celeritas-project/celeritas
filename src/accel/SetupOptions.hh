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

    //!@{
    //! \name Celeritas stepper options
    //! Number of track "slots" to be transported simultaneously
    size_type max_num_tracks{};
    //! Maximum number of events in use
    size_type max_num_events{};
    //! Limit on number of step iterations before aborting
    size_type max_steps = no_max_steps();
    //! Maximum number of track initializers (primaries+secondaries)
    size_type initializer_capacity{};
    //! At least the average number of secondaries per track slot
    real_type secondary_stack_factor{};
    //! Sync the GPU at every kernel for error checking
    bool sync{false};
    //!@}

    //!@{
    //! \name CUDA options
    size_type cuda_stack_size{};
    size_type cuda_heap_size{};
    //!@}
};

//---------------------------------------------------------------------------//
} // namespace celeritas
