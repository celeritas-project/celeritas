//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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

#include "corecel/sys/Device.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/ActionInterface.hh"

class G4LogicalVolume;

namespace celeritas
{
namespace inp
{
struct FrameworkInput;
struct GeantSd;
}

class CoreParams;
struct AlongStepFactoryInput;
//---------------------------------------------------------------------------//
/*!
 * Control options for initializing Celeritas SD callbacks.
 *
 * By default, Celeritas connects to Geant4 sensitive detectors so that it
 * reconstructs full-fidelity hits with all available step information.
 * - If your problem has no SDs, you must set \c enabled to \c false.
 * - By default, steps that do not deposit energy do not generate any hits.
 * - To improve performance and memory usage, determine what quantities (time,
 *   position, direction, touchable, ...) are required by your setup's
 *   sensitive detectors and set all other attributes to \c false.
 * - Reconstructing the full geometry status using \c locate_touchable is the
 *   most expensive detector option. Disable it unless your SDs require (e.g.)
 *   the volume's copy number to locate a detector submodule.
 * - Some reconstructed track attributes (such as post-step material) are
 *   currently never set because they are rarely used in practice. Contact the
 *   Celeritas team or submit a pull request to add this functionality.
 *
 * Various attributes on the step, track, and pre/post step points may be
 * available depending on the selected options.
 * - Disabling \c track will leave \c G4Step::GetTrack as \c nullptr .
 * - Enabling \c track will set the \c Charge attribute on the
 *   pre-step.
 * - Requested post-step data including \c GlobalTime, \c Position, \c
 *   KineticEnergy, and \c MomentumDirection will be copied to the \c Track
 *   when the combination of options is enabled.
 * - Some pre-step properties (\c Material and \c MaterialCutsCouple, and
 *   sensitive detector) are always updated. Post-step values for those are not
 *   set.
 * - Track and Parent IDs will \em never be a valid value since Celeritas track
 *   counters are independent from Geant4 track counters. Similarly, special
 *   Geant4 user-defined \c UserInformation and \c AuxiliaryTrackInformation
 *   are never set.
 *
 * The \c force_volumes option can be used for unusual cases (i.e., when using
 * a custom run manager) that do not define SDs on the "master" thread.
 * Similarly, the \c skip_volumes option allows optimized GPU-defined SDs to be
 * used in place of a Geant4 callback. For both options, the \c
 * FindVolumes helper function can be used to determine LV pointers from
 * the volume names.
 *
 * \note These setup options affect only the \c GeantSd construction that is
 * responsible for reconstructing CPU hits and sending directly to the Geant4
 * detectors. It does not change the underlying physics.
 *
 * \note This class will be replaced in v1.0
 *       by \c celeritas::inp::SensitiveDetector .
 */
struct SDSetupOptions
{
    struct StepPoint
    {
        bool global_time{true};
        bool position{true};
        bool direction{true};  //!< AKA momentum direction
        bool kinetic_energy{true};
    };

    //! Call back to Geant4 sensitive detectors
    bool enabled{true};
    //! Skip steps that do not deposit energy locally
    bool ignore_zero_deposition{true};
    //! Save energy deposition
    bool energy_deposition{true};
    //! Save physical step length
    bool step_length{true};
    //! Set TouchableHandle for PreStepPoint
    bool locate_touchable{true};
    //! Set TouchableHandle for PostStepPoint
    bool locate_touchable_post{true};
    //! Create a track with the dynamic particle type and post-step data
    bool track{true};
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
 *
 * \note This class will be replaced in v1.0
 *       by \c celeritas::inp::FrameworkInput .
 */
struct SetupOptions
{
    //!@{
    //! \name Type aliases
    using size_type = unsigned int;
    using real_type = double;

    using SPConstAction = std::shared_ptr<CoreStepActionInterface const>;
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
    //! Filename for JSON diagnostic output, empty to disable
    std::string output_file{"celeritas.out.json"};
    //! Filename for ROOT dump of physics data
    std::string physics_output_file;
    //! Filename to dump a ROOT/HepMC3 copy of offloaded tracks as events
    std::string offload_output_file;
    //! Filename to dump a GDML file for debugging inside frameworks
    std::string geometry_output_file;
    //!@}

    //!@{
    //! \name Celeritas stepper options

    //! Number of track "slots" to be transported simultaneously
    size_type max_num_tracks{};
    //! Maximum number of events in use (DEPRECATED: remove in v0.7)
    size_type max_num_events{};
    //! Limit on number of steps per track before killing
    size_type max_steps = no_max_steps();
    //! Limit on number of step iterations before aborting
    size_type max_step_iters = no_max_steps();
    //! Maximum number of track initializers (primaries+secondaries)
    size_type initializer_capacity{};
    //! At least the average number of secondaries per track slot
    real_type secondary_stack_factor{3.0};
    //! Number of tracks to buffer before offloading (if unset: max num tracks)
    size_type auto_flush{};
    //!@}

    //!@{
    //! \name Track reordering options
    TrackOrder track_order{TrackOrder::size_};
    //!@}

    //! Set the number of streams (defaults to run manager # threads)
    IntAccessor get_num_streams;

    //!@{
    //! \name Stepping actions

    AlongStepFactory make_along_step;
    //!@}

    //!@{
    //! \name Field options

    size_type max_field_substeps{10};
    //!@}

    //! Sensitive detector options
    SDSetupOptions sd;

    //!@{
    //! \name Physics options

    //! Do not use Celeritas physics for the given Geant4 process names
    VecString ignore_processes;
    //!@}

    //!@{
    //! \name CUDA options

    //! Per-thread stack size (may be needed for VecGeom) [B]
    size_type cuda_stack_size{};
    //! Dynamic heap size (may be needed for VecGeom) [B]
    size_type cuda_heap_size{};
    //! Sync the GPU at every kernel for timing
    bool action_times{false};
    //! Launch all kernels on the default stream for debugging (REMOVED)
    bool default_stream{false};
    //!@}

    //!@{
    //! \name Diagnostic setup

    //! Filename base for slot diagnostics
    std::string slot_diagnostic_prefix;

    //! Add additional diagnostic user actions [EXPERIMENTAL]
    std::function<void(CoreParams const&)> add_user_actions;

    //!@}

    explicit inline operator bool() const
    {
        return max_num_tracks > 0 && static_cast<bool>(make_along_step);
    }
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Find volumes by name for SDSetupOptions
std::unordered_set<G4LogicalVolume const*>
    FindVolumes(std::unordered_set<std::string>);

// Convert SD options for forward compatibility
inp::GeantSd to_inp(SDSetupOptions const& so);

// Construct a framework input
inp::FrameworkInput to_inp(SetupOptions const& so);

//---------------------------------------------------------------------------//
}  // namespace celeritas
