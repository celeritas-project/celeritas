//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/DielectricInteractionModel.cc
//---------------------------------------------------------------------------//
#include "DielectricInteractionModel.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/grid/NonuniformGridBuilder.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "DielectricInteractionExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct the model from an ID and a layer map.
 */
DielectricInteractionModel::DielectricInteractionModel(
    SurfaceModelId id, std::map<PhysSurfaceId, InputT> const& layer_map)
    : SurfaceModel(id, "interaction-dielectric")
{
    HostVal<DielectricData> dielectric_data;
    HostVal<UnifiedReflectionData> reflection_data;

    auto build_dielectric = make_builder(&dielectric_data.interface);
    auto build_spec_spike = make_builder(&reflection_data.specular_spike);
    auto build_spec_lobe = make_builder(&reflection_data.specular_lobe);
    auto build_back_scatter = make_builder(&reflection_data.back_scattering);
    NonuniformGridBuilder build_grid(&reflection_data.reals);

    surfaces_.reserve(layer_map.size());

    for (auto const& [surface, input] : layer_map)
    {
        surfaces_.push_back(surface);

        build_dielectric.push_back(
            static_cast<DielectricInterface>(input.is_metal));

        build_spec_spike.push_back(build_grid(input.reflection.specular_spike));
        build_spec_lobe.push_back(build_grid(input.reflection.specular_lobe));
        build_back_scatter.push_back(build_grid(input.reflection.backscatter));
    }

    CELER_ENSURE(dielectric_data);
    CELER_ENSURE(reflection_data);

    dielectric_data_
        = CollectionMirror<DielectricData>{std::move(dielectric_data)};
    reflection_data_
        = CollectionMirror<UnifiedReflectionData>{std::move(reflection_data)};
}

//---------------------------------------------------------------------------//
/*!
 * Execute model with host data.
 */
void DielectricInteractionModel::step(CoreParams const& params,
                                      CoreStateHost& state) const
{
    launch_action(
        state,
        make_surface_physics_executor(
            params.ptr<MemSpace::native>(),
            state.ptr(),
            SurfacePhysicsOrder::interaction,
            this->surface_model_id(),
            DielectricInteractionExecutor{dielectric_data_.host_ref(),
                                          reflection_data_.host_ref()}));
}

//---------------------------------------------------------------------------//
/*!
 * Execute model with device data.
 */
#if !CELER_USE_DEVICE
void DielectricInteractionModel::step(CoreParams const&, CoreStateHost&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
