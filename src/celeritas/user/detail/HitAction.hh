//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/HitAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/data/CollectionMirror.hh"
#include "celeritas/global/ActionInterface.hh"

#include "../HitData.hh"
#include "../HitInterface.hh"
#include "HitBuffer.hh"
#include "celeritas_config.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Gather track step properties at a point during the step.
 *
 * This implementation class is constructed by the HitCollector.
 */
class HitAction final : public ExplicitActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPHitInterface = std::shared_ptr<HitInterface>;
    using SPHitBuffer    = std::shared_ptr<HitBuffer>;
    //!@}

  public:
    // Construct with next action ID
    explicit HitAction(ActionId            id,
                       ActionOrder         order,
                       SPHitInterface      callback,
                       const HitSelection& selection,
                       SPHitBuffer         buffer);

    // Launch kernel with host data
    void execute(CoreHostRef const&) const final;

    // Launch kernel with device data
    void execute(CoreDeviceRef const&) const final;

    //! ID of the model
    ActionId action_id() const final { return id_; }

    // Short name for the action
    std::string label() const final;

    // Name of the action (for user output)
    std::string description() const final;

    //! Dependency ordering of the action
    ActionOrder order() const final { return ActionOrder::along; }

  private:
    //// DATA ////

    ActionId                        id_;
    ActionOrder                     order_;
    CollectionMirror<HitParamsData> params_;
    SPHitInterface                  callback_;
    SPHitBuffer                     buffer_;

    //// HELPER FUNCTIONS ////

    template<MemSpace M>
    inline HitStateData<M, Ownership::reference>
    get_state(const CoreRef<M>& core_data) const;
};

//---------------------------------------------------------------------------//
// PRIVATE HELPER FUNCTIONS
//---------------------------------------------------------------------------//
template<MemSpace M>
HitStateData<M, Ownership::reference>
HitAction::get_state(const CoreRef<M>& core_data) const
{
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
