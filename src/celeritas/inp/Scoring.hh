//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Scoring.hh
//---------------------------------------------------------------------------//
#pragma once

#include <optional>
#include <unordered_set>
#include <variant>
#include <vector>

#include "corecel/io/Label.hh"

class G4LogicalVolume;

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
//! Options for saving attributes at each step point
struct GeantSdStepPointAttributes
{
    //! Store the time since the start of the event
    bool global_time{true};
    //! Store the step point position
    bool position{true};
    //! Store the step point direction (AKA momentum direction)
    bool direction{true};
    //! Store the step point energy
    bool kinetic_energy{true};
};

//---------------------------------------------------------------------------//
/*!
 * Control options for Geant4 sensitive detector integration.
 *
 * By default, Celeritas connects to Geant4 sensitive detectors so that it
 * reconstructs full-fidelity hits with all available step information.
 *
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
 *
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
 * \todo For improved granularity in models with duplicate names, we could add
 * a vector of \c Label to \c VariantSetVolume .
 * \todo change from \c unordered_set to \c set for better reproducibility in
 * serialized output
 *
 * \sa celeritas::GeantSd
 */
struct GeantSd
{
    //! Provide either a set of labels or a set of pointers to Geant4 objects
    using SetVolume = std::unordered_set<G4LogicalVolume const*>;
    using SetString = std::unordered_set<std::string>;
    using VariantSetVolume = std::variant<SetVolume, SetString>;

    //! Skip steps that do not deposit energy locally
    bool ignore_zero_deposition{true};
    //! Save energy deposition
    bool energy_deposition{true};
    //! Save physical step length
    bool step_length{true};
    //! Set TouchableHandle for PreStepPoint
    bool locate_touchable{true};
    //! Create a track with the dynamic particle type and post-step data
    bool track{true};

    //! Options for saving and converting beginning-of-step data
    GeantSdStepPointAttributes pre;
    //! Options for saving and converting end-of-step data
    GeantSdStepPointAttributes post;

    //! Manually list LVs that don't have an SD on the master thread
    VariantSetVolume force_volumes;
    //! List LVs that should *not* have automatic hit mapping
    VariantSetVolume skip_volumes;
};

//---------------------------------------------------------------------------//
/*!
 * Integrate energy deposition in each volume over all events.
 *
 * \sa celeritas::SimpleCalo.
 */
struct SimpleCalo
{
    //! List of geometry volumes to score
    std::vector<Label> volumes;
};

//---------------------------------------------------------------------------//
/*!
 * Enable scoring of hits or other quantities.
 *
 * If the problem to be executed has no sensitive detectors, \c sd must be
 * \c std::nullopt (unset).
 */
struct Scoring
{
    //! Enable Geant4 sensitive detector integration
    std::optional<GeantSd> sd;

    //! Add simple on-device calorimeters integrated over events
    std::optional<SimpleCalo> simple_calo;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
