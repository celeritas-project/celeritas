//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/RoughnessModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/surface/SurfaceModel.hh"

#include "GaussianRoughnessModel.hh"
#include "PolishedRoughnessModel.hh"
#include "RoughnessApplier.hh"
#include "SmearRoughnessModel.hh"

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
    RoughnessModel ...;
   \endcode
 */
template<class Controller>
class RoughnessModel : public SurfaceModel
{
  public:
    template<class T>
    static std::shared_ptr<RoughnessModel<Controller>>
    build_model(SurfaceModelId model_id,
                std::map<PhysSurfaceId, T> const& layer_map)
    {
        std::vector<PhysSurfaceId> surfaces;
        std::vector<T> inputs;

        for (auto const& [surface, input] : layer_map)
        {
            CELER_ENSURE(surface);
            surfaces.push_back(surface);
            inputs.push_back(input);
        }

        CELER_ENSURE(surfaces.size() == layer_map.size());
        CELER_ENSURE(inputs.size() == layer_map.size());

        return std::make_shared<RoughnessModel>(model_id,
                                                Controller::label,
                                                std::move(surfaces),
                                                Controller{inputs});
    }

    std::vector<PhysSurfaceId> get_surfaces() const final { return surfaces_; }

    void step(CoreParams const& params, CoreStateHost& state) const final;

    void step(CoreParams const&, CoreStateDevice&) const final;

    RoughnessModel(SurfaceModelId model_id,
                   std::string_view label,
                   std::vector<PhysSurfaceId> surfaces,
                   Controller controller)
        : SurfaceModel(model_id, label)
        , controller_(std::move(controller))
        , surfaces_(std::move(surfaces))
    {
    }

  protected:
    Controller controller_;
    std::vector<PhysSurfaceId> surfaces_;
};

extern template class RoughnessModel<SmearRoughnessModelController>;
extern template class RoughnessModel<PolishedRoughnessModelController>;
extern template class RoughnessModel<GaussianRoughnessModelController>;

using SmearRoughnessModel = RoughnessModel<SmearRoughnessModelController>;
using PolishedRoughnessModel = RoughnessModel<PolishedRoughnessModelController>;
using GaussianRoughnessModel = RoughnessModel<GaussianRoughnessModelController>;

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
