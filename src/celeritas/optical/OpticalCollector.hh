//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalCollector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/data/AuxInterface.hh"
#include "celeritas/Types.hh"

#include "OffloadData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ActionRegistry;
class CoreParams;

namespace optical
{
class CerenkovParams;
class MaterialParams;
class ScintillationParams;
}  // namespace optical

namespace detail
{
class CerenkovOffloadAction;
class CerenkovGeneratorAction;
class OffloadGatherAction;
class OpticalLaunchAction;
class OffloadParams;
class ScintOffloadAction;
class ScintGeneratorAction;
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Generate and track optical photons.
 *
 * This class is the interface between the main stepping loop and the photon
 * stepping loop and constructs kernel actions for:
 * - gathering the pre-step data needed to generate the optical distributions,
 * - generating the scintillation and Cerenkov optical distributions at the
 *   end of the step, and
 * - launching the photon stepping loop.
 *
 * The photon stepping loop will then generate optical primaries.
 *
 * The "collector" (TODO: rename?) will "own" the optical state data and
 * optical params since it's the only thing that launches the optical stepping
 * loop.
 *
 * \todo Rename to OpticalOffload
 */
class OpticalCollector
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstCerenkov = std::shared_ptr<optical::CerenkovParams const>;
    using SPConstMaterial = std::shared_ptr<optical::MaterialParams const>;
    using SPConstScintillation
        = std::shared_ptr<optical::ScintillationParams const>;
    //!@}

    struct Input
    {
        //! Optical physics material for materials
        SPConstMaterial material;
        SPConstCerenkov cerenkov;
        SPConstScintillation scintillation;

        //! Number track slots in the optical loop
        size_type num_track_slots{};

        //! Number of steps that have created optical particles
        size_type buffer_capacity{};

        //! Maximum number of buffered initializers in optical tracking loop
        size_type initializer_capacity{};

        //! Threshold number of initializers for launching optical loop
        size_type auto_flush{};

        //! True if all input is assigned and valid
        explicit operator bool() const
        {
            return material && (scintillation || cerenkov)
                   && num_track_slots > 0 && buffer_capacity > 0
                   && initializer_capacity > 0 && auto_flush > 0;
        }
    };

  public:
    // Construct with core data and optical params
    OpticalCollector(CoreParams const&, Input&&);

    // Aux ID for optical offload data
    AuxId offload_aux_id() const;

    // Aux ID for optical state data
    AuxId optical_aux_id() const;

  private:
    //// TYPES ////

    using SPOffloadParams = std::shared_ptr<detail::OffloadParams>;
    using SPCerenkovAction = std::shared_ptr<detail::CerenkovOffloadAction>;
    using SPScintAction = std::shared_ptr<detail::ScintOffloadAction>;
    using SPGatherAction = std::shared_ptr<detail::OffloadGatherAction>;
    using SPCerenkovGenAction
        = std::shared_ptr<detail::CerenkovGeneratorAction>;
    using SPScintGenAction = std::shared_ptr<detail::ScintGeneratorAction>;
    using SPLaunchAction = std::shared_ptr<detail::OpticalLaunchAction>;

    //// DATA ////

    SPOffloadParams offload_params_;

    SPGatherAction gather_action_;
    SPCerenkovAction cerenkov_action_;
    SPScintAction scint_action_;
    SPCerenkovGenAction cerenkov_gen_action_;
    SPScintGenAction scint_gen_action_;
    SPLaunchAction launch_action_;

    // TODO: tracking loop launch action
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
