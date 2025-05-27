//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalCollector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/data/AuxInterface.hh"
#include "corecel/data/AuxStateData.hh"
#include "corecel/data/AuxStateVec.hh"
#include "celeritas/Types.hh"

#include "CoreState.hh"
#include "Model.hh"
#include "gen/OffloadData.hh"
#include "gen/detail/GeneratorAction.hh"

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
class CherenkovOffloadAction;
class OffloadGatherAction;
class OpticalLaunchAction;
class ScintOffloadAction;
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
            return material && (scintillation || cherenkov)
                   && num_track_slots > 0 && buffer_capacity > 0
                   && initializer_capacity > 0 && auto_flush > 0
                   && !model_builders.empty();
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
    template<MemSpace M>
    inline OpticalAccumStats exchange_counters(AuxStateVec& aux) const;

    // Get queued buffer sizes
    template<MemSpace M>
    inline OpticalBufferSize buffer_counts(AuxStateVec const& aux) const;

  private:
    //// TYPES ////

    using SPCherenkovOffload = std::shared_ptr<detail::CherenkovOffloadAction>;
    using SPScintOffload = std::shared_ptr<detail::ScintOffloadAction>;
    using SPGatherAction = std::shared_ptr<detail::OffloadGatherAction>;
    using CherenkovGenAction
        = detail::GeneratorAction<CherenkovData, CherenkovGenerator>;
    using ScintGenAction
        = detail::GeneratorAction<ScintillationData, ScintillationGenerator>;
    using SPCherenkovGen = std::shared_ptr<CherenkovGenAction>;
    using SPScintGen = std::shared_ptr<ScintGenAction>;
    using SPLaunchAction = std::shared_ptr<detail::OpticalLaunchAction>;

    //// DATA ////

    SPGatherAction gather_;
    SPCherenkovOffload cherenkov_offload_;
    SPScintOffload scint_offload_;
    SPCherenkovGen cherenkov_gen_;
    SPScintGen scint_gen_;
    SPLaunchAction launch_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get and reset cumulative statistics on optical generation from a state.
 */
template<MemSpace M>
OpticalAccumStats OpticalCollector::exchange_counters(AuxStateVec& aux) const
{
    auto& state = get<optical::CoreState<M>>(aux, this->optical_aux_id());
    auto& accum = state.accum();

    if (auto id = this->cherenkov_aux_id())
    {
        auto& gen = dynamic_cast<GeneratorStateBase const&>(aux.at(id));
        accum.cherenkov = gen.accum;
    }
    if (auto id = this->scintillation_aux_id())
    {
        auto& gen = dynamic_cast<GeneratorStateBase const&>(aux.at(id));
        accum.scintillation = gen.accum;
    }

    return std::exchange(accum, {});
}

//---------------------------------------------------------------------------//
/*!
 * Get info on the number of tracks in the buffer.
 */
template<MemSpace M>
auto OpticalCollector::buffer_counts(AuxStateVec const& aux) const
    -> OpticalBufferSize
{
    OpticalBufferSize result;

    auto const& state = get<optical::CoreState<M>>(aux, this->optical_aux_id());
    result.photons = state.counters().num_pending;

    if (auto id = this->cherenkov_aux_id())
    {
        auto& gen = dynamic_cast<GeneratorStateBase const&>(aux.at(id));
        result.distributions += gen.buffer_size;
    }
    if (auto id = this->scintillation_aux_id())
    {
        auto& gen = dynamic_cast<GeneratorStateBase const&>(aux.at(id));
        result.distributions += gen.buffer_size;
    }

    return result;
}
//---------------------------------------------------------------------------//
}  // namespace celeritas
