//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ActionDiagnostic.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <memory>
#include <vector>

#include "corecel/data/StreamStore.hh"
#include "corecel/io/OutputInterface.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/CoreTrackData.hh"

#include "ParticleTallyData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ParticleParams;
class ActionRegistry;

//---------------------------------------------------------------------------//
/*!
 * Tally post-step actions for each particle type.
 */
class ActionDiagnostic final : public ExplicitActionInterface,
                               public BeginRunActionInterface,
                               public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using WPConstActionRegistry = std::weak_ptr<ActionRegistry const>;
    using WPConstParticle = std::weak_ptr<ParticleParams const>;
    using MapStringCount = std::map<std::string, size_type>;
    using VecCount = std::vector<size_type>;
    using VecVecCount = std::vector<VecCount>;
    //!@}

  public:
    // Construct with ID, deferring other data till later
    explicit ActionDiagnostic(ActionId id);

    // Default destructor
    ~ActionDiagnostic();

    //!@{
    //! \name Action interface
    //! ID of the action
    ActionId action_id() const final { return id_; }
    //! Short name for the action
    std::string label() const final { return "action-diagnostic"; }
    // Description of the action for user interaction
    std::string description() const final;
    //! Dependency ordering of the action
    ActionOrder order() const final { return ActionOrder::post; }
    //!@}

    //!@{
    //! \name BeginRunAction interface
    // Set host data at the beginning of a run
    void begin_run(CoreParams const&, CoreStateHost&) final;
    // Set device data at the beginning of a run
    void begin_run(CoreParams const&, CoreStateDevice&) final;
    //!@}

    //!@{
    //! \name ExplicitAction interface
    // Launch kernel with host data
    void execute(CoreParams const&, CoreStateHost&) const final;
    // Launch kernel with device data
    void execute(CoreParams const&, CoreStateDevice&) const final;
    //!@}

    //!@{
    //! \name Output interface
    //! Category of data to write
    Category category() const final { return Category::result; }
    // Write output to the given JSON object
    void output(JsonPimpl*) const final;
    //!@}

    // Get the nonzero diagnostic results accumulated over all streams
    MapStringCount calc_actions_map() const;

    // Get the diagnostic results accumulated over all streams
    VecVecCount calc_actions() const;

    // Diagnostic state data size (number of particles times number of actions)
    size_type state_size() const;

    // Reset diagnostic results
    void clear();

  private:
    using StoreT = StreamStore<ParticleTallyParamsData, ParticleTallyStateData>;

    ActionId id_;

    WPConstActionRegistry action_reg_;
    WPConstParticle particle_;

    mutable StoreT store_;

    //// HELPER METHODS ////

    // Build the storage for diagnostic parameters and stream-dependent states
    void begin_run_impl(CoreParams const&);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
