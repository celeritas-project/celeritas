//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ActionDiagnostic.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <vector>

#include "corecel/data/StreamStore.hh"
#include "corecel/io/OutputInterface.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/CoreTrackData.hh"

#include "ActionDiagnosticData.hh"

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
                               public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstActionRegistry = std::shared_ptr<ActionRegistry const>;
    using SPConstParticle = std::shared_ptr<ParticleParams const>;
    using MapStringCount = std::map<std::string, size_type>;
    using VecCount = std::vector<size_type>;
    using VecVecCount = std::vector<VecCount>;
    //!@}

  public:
    //! Construct with action registry and particle data
    ActionDiagnostic(ActionId id,
                     SPConstActionRegistry action_reg,
                     SPConstParticle particle,
                     size_type num_streams);

    //! Default destructor
    ~ActionDiagnostic();

    //!@{
    //! \name ExplicitAction interface
    // Execute action with host data
    void execute(ParamsHostCRef const&, StateHostRef&) const final;
    // Execute action with device data
    void execute(ParamsDeviceCRef const&, StateDeviceRef&) const final;
    //! ID of the action
    ActionId action_id() const final { return id_; }
    //! Short name for the action
    std::string label() const final { return "action-diagnostic"; }
    //! Description of the action for user interaction
    std::string description() const final { return "Action diagnostic"; }
    //! Dependency ordering of the action
    ActionOrder order() const final { return ActionOrder::post; }
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

    // Number of tally bins (number of particles times number of actions)
    size_type num_bins() const;

    // Reset diagnostic results
    void clear();

  private:
    using StoreT
        = StreamStore<ActionDiagnosticParamsData, ActionDiagnosticStateData>;
    using UPStoreT = std::unique_ptr<StoreT>;

    ActionId id_;
    SPConstActionRegistry action_reg_;
    SPConstParticle particle_;
    size_type num_streams_;
    UPStoreT store_;

    //// HELPER METHODS ////

    // Build the storage for diagnostic parameters and stream-dependent states
    void build_stream_store() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

#if !CELER_USE_DEVICE
inline void
ActionDiagnostic::execute(ParamsDeviceCRef const&, StateDeviceRef&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
