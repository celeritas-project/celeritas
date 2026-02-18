//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/FresnelReflectivityModel.cc
//---------------------------------------------------------------------------//
#include "FresnelReflectivityModel.hh"

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "FresnelReflectivityExecutor.hh"
#include "ReflectivityApplier.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct the model from an ID and a layer map.
 */
FresnelReflectivityModel::FresnelReflectivityModel(
    SurfaceModelId id, std::map<PhysSurfaceId, InputT> const& layer_map)
    : SurfaceModel(id, "reflectivity-fresnel")
{
    surfaces_.reserve(layer_map.size());

    for ([[maybe_unused]] auto const& [surface, _] : layer_map)
    {
        surfaces_.push_back(surface);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Execute the model with host data.
 */
void FresnelReflectivityModel::step(CoreParams const& params,
                                    CoreStateHost& state) const
{
    launch_action(state,
                  make_surface_physics_executor(
                      params.ptr<MemSpace::native>(),
                      state.ptr(),
                      SurfacePhysicsOrder::reflectivity,
                      this->surface_model_id(),
                      ReflectivityApplier{FresnelReflectivityExecutor{}}));
}

//---------------------------------------------------------------------------//
/*!
 * Execute the model with device data.
 */
#if !CELER_USE_DEVICE
void FresnelReflectivityModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
