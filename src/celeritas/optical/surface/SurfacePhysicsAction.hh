//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/action/ActionInterface.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    SurfacePhysicsAction ...;
   \endcode
 */
class SurfacePhysicsAction final : public StaticConcreteAction,
                                   public OpticalStepActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPModel = std::shared_ptr<SurfaceModel>;
    using VecModel = std::vector<SPModel>;
    //!@}

  public:
    // Construct with defaults
    inline SurfacePhysicsAction(ActionId);

    StepActionOrder order() const final { return StepActionOrder::post; }

    void step(CoreParams const& params, CoreStateHost& state) final;
    void step(CoreParams const& params, CoreStateDevice& state) final;

  private:
    template<MemSpace M>
    void step_impl(CoreParams const& params, CoreState<M>& state);

    VecModel roughness_models_;
    VecModel reflectivity_models_;
    VecModel interaction_models_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

template<MemSpace M>
void SurfacePhysicsAction::step_impl(CoreParams const& params,
                                     CoreState<M>& state)
{
    for (auto& model : roughness_models_)
    {
        model->step(params, state);
    }

    for (auto& model : reflectivity_models_)
    {
        model->step(params, state);
    }

    for (auto& model : interaction_models_)
    {
        model->step(params, state);
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
