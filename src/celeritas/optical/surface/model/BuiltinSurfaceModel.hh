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
 * Templated base class for built-in optical surface physics models.
 *
 * Built-in surface physics models share a common format for their input data
 * as well as the parameters to pass to their kernel launchers. The \c
 * Applier wrappers executor calls to factor out common behavior between
 * surface physics steps.
 */
template<SurfacePhysicsOrder S, template<class> class Applier>
class BuiltinSurfaceModel : public SurfaceModel
{
  public:
    //! Get surfaces handled by this model.
    std::vector<PhysSurfaceId> const& get_surfaces() const final
    {
        return surfaces_;
    }

  protected:
    // Construct from ID, label, and surfaces
    template<class InputT>
    BuiltinSurfaceModel(SurfaceModelId model_id,
                        std::string_view label,
                        std::map<PhysSurfaceId, InputT> const& layer_map);

    // Construct executor with applier wrapper
    template<MemSpace M, class E>
    inline auto make_executor(CoreParams const&, CoreState<M>&, E&&) const;

  private:
    std::vector<PhysSurfaceId> surfaces_;
};
//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from ID, label, and surfaces.
 */
template<SurfacePhysicsOrder S, template<class> class Applier>
template<class InputT>
BuiltinSurfaceModel<S, Applier>::BuiltinSurfaceModel(
    SurfaceModelId model_id,
    std::string_view label,
    std::map<PhysSurfaceId, InputT> const& layer_map)
    : SurfaceModel(model_id, label)
{
    surfaces_.reserve(layer_map.size());

    for (auto const& [surface, input] : layer_map)
    {
        CELER_ENSURE(surface);
        surfaces_.push_back(surface);
    }

    CELER_ENSURE(layer_map.size() == surfaces_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Construct a surface physics executor with built-in applier wrapper.
 */
template<SurfacePhysicsOrder S, template<class> class Applier>
template<MemSpace M, class E>
auto BuiltinSurfaceModel<S, Applier>::make_executor(CoreParams const& params,
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
// TEMPLATE ALIASES
//---------------------------------------------------------------------------//

using BuiltinRoughnessModel
    = BuiltinSurfaceModel<SurfacePhysicsOrder::roughness, RoughnessApplier>;
using BuiltinReflectivityModel
    = BuiltinSurfaceModel<SurfacePhysicsOrder::reflectivity, TrivialApplier>;
using BuiltinInteractionModel
    = BuiltinSurfaceModel<SurfacePhysicsOrder::interaction, TrivialApplier>;

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
