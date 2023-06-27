//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SetupOptions.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "celeritas/Types.hh"

class G4LogicalVolume;

namespace celeritas
{
struct AlongStepFactoryInput;
class ExplicitActionInterface;
//---------------------------------------------------------------------------//
/*!
 * Control options for initializing Celeritas SD callbacks.
 *
 * These affect only the \c HitManager construction that is responsible for
 * reconstructing CPU hits and sending directly to the Geant4 detectors.
 */
struct SDSetupOptions
{
    struct StepPoint
    {
        bool global_time{false};
        bool position{false};
        bool kinetic_energy{false};
    };

    //! Call back to Geant4 sensitive detectors
    bool enabled{false};
    //! Skip steps that do not deposit energy locally
    bool ignore_zero_deposition{true};
    //! Save energy deposition
    bool energy_deposition{true};
    //! Set TouchableHandle for PreStepPoint
    bool locate_touchable{false};
    //! Options for saving and converting beginning-of-step data
    StepPoint pre;
    //! Options for saving and converting end-of-step data
    StepPoint post;

    //! Manually list LVs that don't have an SD on the master thread
    std::unordered_set<G4LogicalVolume const*> force_volumes;
    //! List LVs that should *not* have automatic hit mapping
    std::unordered_set<G4LogicalVolume const*> skip_volumes;

    //! True if SD is enabled
    explicit operator bool() const { return this->enabled; }
};

//---------------------------------------------------------------------------//
/*!
 * Control options for initializing Celeritas.
 *
 * The interface for the "along-step factory" (input parameters and output) is
 * described in \c AlongStepFactoryInterface .
 */
struct SetupOptions
{
    //!@{
    //! \name Type aliases
    using size_type = unsigned int;
    using real_type = double;

    using SPConstAction = std::shared_ptr<ExplicitActionInterface const>;
    using AlongStepFactory
        = std::function<SPConstAction(AlongStepFactoryInput const&)>;
    using IntAccessor = std::function<int()>;
    using VecString = std::vector<std::string>;
    //!@}

    //! Don't limit the number of steps
    static constexpr size_type no_max_steps()
    {
        return static_cast<size_type>(-1);
    }

    //!@{
    //! \name I/O
    //! GDML filename (optional: defaults to exporting existing Geant4)
    std::string geometry_file;
    //! Filename for JSON diagnostic output
    std::string output_file;
    //! Filename for ROOT dump of physics data
    std::string physics_output_file;
    //!@}

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
    real_type secondary_stack_factor{3.0};
    //!@}

    //! Set the number of streams (defaults to run manager # threads)
    IntAccessor get_num_streams;

    //!@{
    //! \name Stepping actions
    AlongStepFactory make_along_step;
    //!@}

    //!@{
    //! \name Sensitive detector options
    SDSetupOptions sd;
    //!@}

    //!@{
    //! \name Physics options
    //! Ignore the following EM process names
    VecString ignore_processes;
    //!@}

    //!@{
    //! \name CUDA options
    size_type cuda_stack_size{};
    size_type cuda_heap_size{};
    //! Sync the GPU at every kernel for timing
    bool sync{false};
    //! Launch all kernels on the default stream
    bool default_stream{false};
    //!@}

    //!@{
    //! \name Track init options
    TrackOrder track_order{TrackOrder::unsorted};
    //!@}
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Find volumes by name for SDSetupOptions
std::unordered_set<G4LogicalVolume const*>
    FindVolumes(std::unordered_set<std::string>);

//---------------------------------------------------------------------------//
}  // namespace celeritas
