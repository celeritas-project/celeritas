//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/OffloadAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Macros.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/global/ActionInterface.hh"

#include "GeneratorData.hh"

#include "detail/OffloadTraits.hh"

namespace celeritas
{
namespace optical
{
class MaterialParams;
}  // namespace optical

//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 */
template<GeneratorType G>
class OffloadAction final : public CoreStepActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using TraitsT = detail::OffloadTraits<G>;
    using SPConstParams = std::shared_ptr<typename TraitsT::Params const>;
    using SPConstMaterial = std::shared_ptr<optical::MaterialParams const>;
    //!@}

    //! Offload input data
    struct Input
    {
        AuxId step_id;
        AuxId gen_id;
        AuxId optical_id;
        SPConstMaterial material;
        SPConstParams shared;

        explicit operator bool() const
        {
            return step_id && gen_id && optical_id && material && shared;
        }
    };

  public:
    // Construct and add to core params
    static std::shared_ptr<OffloadAction>
    make_and_insert(CoreParams const&, Input&&);

    // Construct with action ID, aux IDs, and optical properties
    OffloadAction(ActionId action_id, Input&&);

    //!@{
    //! \name Action interface

    //! ID of the action
    ActionId action_id() const final { return action_id_; }
    //! Short name for the action
    std::string_view label() const final { return TraitsT::label; }
    //! Description of the action
    std::string_view description() const final { return TraitsT::description; }
    //!@}

    //!@{
    //! \name StepAction interface

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::user_post; }
    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;
    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;
    //!@}

    //// ACCESSORS ////

    //! Access shared data used by this offload physics
    SPConstParams const& params() const { return data_.shared; }

  private:
    //// TYPES ////

    using Executor = typename TraitsT::Executor;

    //// DATA ////

    ActionId action_id_;
    Input data_;

    //// HELPER FUNCTIONS ////

    template<MemSpace M>
    void step_impl(CoreParams const&, CoreState<M>&) const;

    void offload(CoreParams const&, CoreStateHost&) const;
    void offload(CoreParams const&, CoreStateDevice&) const;
};

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

extern template class OffloadAction<GeneratorType::cherenkov>;
extern template class OffloadAction<GeneratorType::scintillation>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
