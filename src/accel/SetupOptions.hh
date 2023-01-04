//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SetupOptions.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <string>

namespace celeritas
{
struct AlongStepFactoryInput;
class ExplicitActionInterface;
//---------------------------------------------------------------------------//
/*!
 * Control options for initializing Celeritas.
 *
 * The interface for the "along-step factory" (input parameters and output) is
 * described in AlongStepFactory.hh .
 */
struct SetupOptions
{
    //!@{
    //! \name Type aliases
    using size_type = unsigned int;
    using real_type = double;

    using SPConstAction = std::shared_ptr<const ExplicitActionInterface>;
    using AlongStepFactory
        = std::function<SPConstAction(const AlongStepFactoryInput&)>;
    //!@}

    //! Don't limit the number of steps
    static constexpr size_type no_max_steps()
    {
        return static_cast<size_type>(-1);
    }

    // TODO: names of sensitive detectors

    //!@{
    //! \name I/O
    //! GDML filename (optional: defaults to exporting existing Geant4)
    std::string geometry_file;
    //! Filename for JSON diagnostic output
    std::string output_file;
    //!@}

    //!@{
    //! \name Celeritas stepper options
    //! Number of track "slots" to be transported simultaneously
    size_type   max_num_tracks{};
    //! Maximum number of events in use
    size_type   max_num_events{};
    //! Limit on number of step iterations before aborting
    size_type   max_steps = no_max_steps();
    //! Maximum number of track initializers (primaries+secondaries)
    size_type   initializer_capacity{};
    //! At least the average number of secondaries per track slot
    real_type   secondary_stack_factor{};
    //! Sync the GPU at every kernel for error checking
    bool sync{false};
    //!@}

    //!@{
    //! \name Stepping actions
    AlongStepFactory make_along_step;
    //!@}

    //!@{
    //! \name CUDA options
    size_type cuda_stack_size{};
    size_type cuda_heap_size{};
    //!@}
};

//---------------------------------------------------------------------------//
} // namespace celeritas
