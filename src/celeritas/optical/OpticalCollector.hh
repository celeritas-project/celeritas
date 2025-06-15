//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalCollector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <optional>

#include "corecel/data/AuxInterface.hh"
#include "corecel/io/Label.hh"
#include "celeritas/Types.hh"

#include "Model.hh"
#include "gen/OffloadData.hh"
#include "gen/detail/GeneratorTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ActionRegistry;
class AuxStateVec;
class CherenkovParams;
class CoreParams;
class ScintillationParams;

namespace optical
{
class MaterialParams;
}  // namespace optical

namespace detail
{
template<GeneratorType G>
class GeneratorAction;
template<GeneratorType G>
class OffloadAction;
class OffloadGatherAction;
class OpticalLaunchAction;
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Generate and track optical photons.
 *
 * This class is the interface between the main stepping loop and the photon
 * stepping loop and constructs kernel actions for:
 * - gathering the pre-step data needed to generate the optical distributions,
 * - generating the scintillation and Cherenkov optical distributions at the
 *   end of the step, and
 * - launching the photon stepping loop.
 *
 * The photon stepping loop will then generate optical primaries.
 *
 * The "collector" (TODO: rename?) will "own" the optical state data and
 * optical params since it's the only thing that launches the optical stepping
 * loop.
 *
 * \todo This doesn't do anything but set up the optical tracking loop: move to
 * \c setup namespace
 */
class OpticalCollector
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstCherenkov = std::shared_ptr<CherenkovParams const>;
    using SPConstMaterial = std::shared_ptr<optical::MaterialParams const>;
    using SPConstScintillation = std::shared_ptr<ScintillationParams const>;
    using OpticalBufferSize = OpticalOffloadCounters<size_type>;
    //!@}

    struct Input
    {
        //! Optical physics models
        std::vector<optical::Model::ModelBuilder> model_builders;

        //! Optical physics material for materials
        SPConstMaterial material;
        SPConstCherenkov cherenkov;
        SPConstScintillation scintillation;

        std::optional<std::vector<Label>> detector_labels;

        //! Number track slots in the optical loop
        size_type num_track_slots{};

        //! Number of steps that have created optical particles
        size_type buffer_capacity{};

        //! Maximum number of buffered initializers in optical tracking loop
        size_type initializer_capacity{};

        //! Threshold number of initializers for launching optical loop
        size_type auto_flush{};

        //! Maximum step iterations before aborting optical loop
        size_type max_step_iters{static_cast<size_type>(-1)};

        //! True if all input is assigned and valid
        explicit operator bool() const
        {
            return material && (scintillation || cherenkov)
                   && num_track_slots > 0 && buffer_capacity > 0
                   && initializer_capacity > 0 && auto_flush > 0
                   && max_step_iters > 0 && !model_builders.empty();
        }
    };

  public:
    // Construct with core data and optical params
    OpticalCollector(CoreParams const&, Input&&);

    // Aux ID for optical Cherenkov offload data
    AuxId cherenkov_aux_id() const;

    // Aux ID for optical scintillation offload data
    AuxId scintillation_aux_id() const;

    // Aux ID for optical state data
    AuxId optical_aux_id() const;

    // Get and reset cumulative statistics on optical tracks from a state
    OpticalAccumStats exchange_counters(AuxStateVec& aux) const;

    // Get queued buffer sizes
    OpticalBufferSize buffer_counts(AuxStateVec const& aux) const;

  private:
    //// TYPES ////

    using GT = detail::GeneratorType;
    template<GT G>
    using GeneratorAction = detail::GeneratorAction<G>;
    template<GT G>
    using OffloadAction = detail::OffloadAction<G>;
    using SPCherenkovOffload = std::shared_ptr<OffloadAction<GT::cherenkov>>;
    using SPScintOffload = std::shared_ptr<OffloadAction<GT::scintillation>>;
    using SPGatherAction = std::shared_ptr<detail::OffloadGatherAction>;
    using SPCherenkovGen = std::shared_ptr<GeneratorAction<GT::cherenkov>>;
    using SPScintGen = std::shared_ptr<GeneratorAction<GT::scintillation>>;
    using SPLaunchAction = std::shared_ptr<detail::OpticalLaunchAction>;

    //// DATA ////

    SPGatherAction gather_;
    SPCherenkovOffload cherenkov_offload_;
    SPScintOffload scint_offload_;
    SPCherenkovGen cherenkov_generate_;
    SPScintGen scint_generate_;
    SPLaunchAction launch_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
