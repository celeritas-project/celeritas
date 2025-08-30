//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/BuiltinSurfaceModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/surface/SurfaceModel.hh"
#include "celeritas/optical/surface/TrackSlotExecutor.hh"

#include "RoughnessApplier.hh"

namespace celeritas
{
namespace optical
{
namespace
{

template<class T>
struct TrivialApplier
{
    T const& executor;

    inline CELER_FUNCTION void operator()(CoreTrackView& track) const
    {
        executor(track);
    }
};

template<SurfacePhysicsOrder>
struct BuiltinApplier
{
    template<class T>
    using Applier = TrivialApplier<T>;
};

template<>
struct BuiltinApplier<SurfacePhysicsOrder::roughness>
{
    template<class T>
    using Applier = RoughnessApplier<T>;
};

}  // namespace
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    BuiltinSurfaceModel ...;
   \endcode
 */
template<SurfacePhysicsOrder S>
class BuiltinSurfaceModel : public SurfaceModel
{
  public:
    template<class T>
    using Applier = typename BuiltinApplier<S>::Applier<T>;

    std::vector<PhysSurfaceId> get_surfaces() const final { return surfaces_; }

  protected:
    BuiltinSurfaceModel(SurfaceModelId model_id,
                        std::string_view label,
                        std::vector<PhysSurfaceId> surfaces)
        : SurfaceModel(model_id, label), surfaces_(std::move(surfaces))
    {
    }

    template<MemSpace M, class E>
    auto
    make_executor(CoreParams const& params, CoreState<M>& state, E&& exec) const
    {
        return make_surface_physics_executor(params.ptr<M>(),
                                             state.ptr(),
                                             S,
                                             this->surface_model_id(),
                                             Applier<E>{std::forward<E>(exec)});
    }

  private:
    std::vector<PhysSurfaceId> surfaces_;
};

template<class T>
std::shared_ptr<T> builtin_model_from_input(
    SurfaceModelId model_id,
    std::map<PhysSurfaceId, typename T::InputT> const& layer_map)
{
    std::vector<PhysSurfaceId> surfaces;
    std::vector<typename T::InputT> inputs;

    for (auto const& [surface, input] : layer_map)
    {
        CELER_ENSURE(surface);
        surfaces.push_back(surface);
        inputs.push_back(input);
    }

    CELER_ENSURE(surfaces.size() == layer_map.size());
    CELER_ENSURE(inputs.size() == layer_map.size());

    return std::make_shared<T>(model_id, std::move(surfaces), inputs);
}

using BuiltinRoughnessModel
    = BuiltinSurfaceModel<SurfacePhysicsOrder::roughness>;
using BuiltinReflectivityModel
    = BuiltinSurfaceModel<SurfacePhysicsOrder::reflectivity>;
using BuiltinInteractionModel
    = BuiltinSurfaceModel<SurfacePhysicsOrder::interaction>;

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
