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
//---------------------------------------------------------------------------//
/*!
 * Trivial applier which just forwards the track to the executor.
 */
template<class T>
struct TrivialApplier
{
    T const& executor;

    inline CELER_FUNCTION void operator()(CoreTrackView& track) const
    {
        executor(track);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Template trait used to select applier wrapper for built-in executors.
 */
template<SurfacePhysicsOrder>
struct BuiltinApplier;

template<>
struct BuiltinApplier<SurfacePhysicsOrder::roughness>
{
    template<class T>
    using Applier = RoughnessApplier<T>;
};

template<>
struct BuiltinApplier<SurfacePhysicsOrder::reflectivity>
{
    template<class T>
    using Applier = TrivialApplier<T>;
};

template<>
struct BuiltinApplier<SurfacePhysicsOrder::interaction>
{
    template<class T>
    using Applier = TrivialApplier<T>;
};

}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Templated base class for built-in optical surface physics models.
 *
 * Built-in surface physics models share a common format for their input data
 * as well as the parameters to pass to their kernel launchers. The \c
 * BuiltinApplier wrappers executor calls to factor out common behavior between
 * surface physics steps.
 */
template<SurfacePhysicsOrder S>
class BuiltinSurfaceModel : public SurfaceModel
{
  public:
    //!@{
    //! \name Type aliases
    template<class T>
    using Applier = typename BuiltinApplier<S>::Applier<T>;
    //!@}

  public:
    //! Get surfaces handled by this model.
    std::vector<PhysSurfaceId> const& get_surfaces() const final
    {
        return surfaces_;
    }

  protected:
    // Construct from ID, label, and surfaces
    BuiltinSurfaceModel(SurfaceModelId model_id,
                        std::string_view label,
                        std::vector<PhysSurfaceId> surfaces);

    // Construct executor with applier wrapper
    template<MemSpace M, class E>
    auto make_executor(CoreParams const&, CoreState<M>&, E&&) const;

  private:
    std::vector<PhysSurfaceId> surfaces_;
};
//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from ID, label, and surfaces.
 */
template<SurfacePhysicsOrder S>
BuiltinSurfaceModel<S>::BuiltinSurfaceModel(SurfaceModelId model_id,
                                            std::string_view label,
                                            std::vector<PhysSurfaceId> surfaces)
    : SurfaceModel(model_id, label), surfaces_(std::move(surfaces))
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct a surface physics executor with built-in applier wrapper.
 */
template<SurfacePhysicsOrder S>
template<MemSpace M, class E>
auto BuiltinSurfaceModel<S>::make_executor(CoreParams const& params,
                                           CoreState<M>& state,
                                           E&& exec) const
{
    return make_surface_physics_executor(params.ptr<M>(),
                                         state.ptr(),
                                         S,
                                         this->surface_model_id(),
                                         Applier<E>{std::forward<E>(exec)});
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Construct built-in model of type \c M from \c inp::SurfacePhysics surface
 * layer map.
 *
 * The model \c M should have a type alias \c M::InputT which corresponds to
 * the input type associated with each surface.
 */
template<class M>
std::shared_ptr<M> builtin_model_from_input(
    SurfaceModelId model_id,
    std::map<PhysSurfaceId, typename M::InputT> const& layer_map)
{
    std::vector<PhysSurfaceId> surfaces;
    std::vector<typename M::InputT> inputs;

    for (auto const& [surface, input] : layer_map)
    {
        CELER_ENSURE(surface);
        surfaces.push_back(surface);
        inputs.push_back(input);
    }

    CELER_ENSURE(surfaces.size() == layer_map.size());
    CELER_ENSURE(inputs.size() == layer_map.size());

    return std::make_shared<M>(model_id, std::move(surfaces), inputs);
}

//---------------------------------------------------------------------------//
// TEMPLATE ALIASES
//---------------------------------------------------------------------------//

using BuiltinRoughnessModel
    = BuiltinSurfaceModel<SurfacePhysicsOrder::roughness>;
using BuiltinReflectivityModel
    = BuiltinSurfaceModel<SurfacePhysicsOrder::reflectivity>;
using BuiltinInteractionModel
    = BuiltinSurfaceModel<SurfacePhysicsOrder::interaction>;

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
