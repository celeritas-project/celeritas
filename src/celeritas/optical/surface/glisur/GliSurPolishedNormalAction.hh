//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/glisur/GliSurPolishedNormalAction.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
class GliSurPolishedNormalAction
    : public OpticalStepActionInterface,
      public ConcreteAction,
      public ParamsDataInterface<GliSurPolishedNormalData>
{
  public:
    using VecPolish = std::vector<real_type>;

    struct Input
    {
        std::vector<VecPolish> model_polishes;  //!< per model polishes
    };

  public:
    GliSurPolishedNormalAction(ActionId, Input const&);

    inline StepActionOrder order() const final
    {
        return StepActionOrder::post;
    }

    void step(CoreParams const&, CoreStateHost&) const final;
    void step(CoreParams const&, CoreStateDevice&) const final;

    HostRef const& host_ref() const final { return data_.host_ref(); }
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    CollectionMirror<GliSurPolishedNormalData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
