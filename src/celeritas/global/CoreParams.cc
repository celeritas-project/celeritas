//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreParams.cc
//---------------------------------------------------------------------------//
#include "CoreParams.hh"

#include <string>
#include <type_traits>
#include <utility>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/data/Ref.hh"
#include "corecel/io/BuildOutput.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputRegistry.hh"  // IWYU pragma: keep
#include "corecel/sys/Device.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/KernelRegistry.hh"
#include "corecel/sys/MemRegistry.hh"
#include "corecel/sys/ScopedMem.hh"
#include "celeritas/geo/GeoMaterialParams.hh"  // IWYU pragma: keep
#include "celeritas/geo/GeoParams.hh"  // IWYU pragma: keep
#include "celeritas/geo/GeoParamsOutput.hh"
#include "celeritas/geo/generated/BoundaryAction.hh"
#include "celeritas/global/ActionRegistryOutput.hh"
#include "celeritas/mat/MaterialParams.hh"  // IWYU pragma: keep
#include "celeritas/mat/MaterialParamsOutput.hh"
#include "celeritas/phys/CutoffParams.hh"  // IWYU pragma: keep
#include "celeritas/phys/ParticleParams.hh"  // IWYU pragma: keep
#include "celeritas/phys/ParticleParamsOutput.hh"
#include "celeritas/phys/PhysicsParams.hh"  // IWYU pragma: keep
#include "celeritas/phys/PhysicsParamsOutput.hh"
#include "celeritas/random/RngParams.hh"  // IWYU pragma: keep
#include "celeritas/track/ExtendFromSecondariesAction.hh"
#include "celeritas/track/InitializeTracksAction.hh"
#include "celeritas/track/SimParams.hh"  // IWYU pragma: keep
#include "celeritas/track/SortTracksAction.hh"
#include "celeritas/track/TrackInitParams.hh"  // IWYU pragma: keep

#include "ActionInterface.hh"
#include "ActionRegistry.hh"  // IWYU pragma: keep
#include "alongstep/AlongStepNeutralAction.hh"

#if CELERITAS_USE_JSON
#    include "corecel/io/OutputInterfaceAdapter.hh"
#    include "corecel/sys/DeviceIO.json.hh"
#    include "corecel/sys/EnvironmentIO.json.hh"
#    include "corecel/sys/KernelRegistryIO.json.hh"
#    include "corecel/sys/MemRegistryIO.json.hh"
#endif

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER CLASSES AND FUNCTIONS
//---------------------------------------------------------------------------//
//!@{
template<MemSpace M>
CoreParamsData<Ownership::const_reference, M>
build_params_refs(CoreParams::Input const& p, CoreScalars const& scalars)
{
    CELER_EXPECT(scalars);

    CoreParamsData<Ownership::const_reference, M> ref;

    ref.scalars = scalars;
    ref.geometry = get_ref<M>(*p.geometry);
    ref.geo_mats = get_ref<M>(*p.geomaterial);
    ref.materials = get_ref<M>(*p.material);
    ref.particles = get_ref<M>(*p.particle);
    ref.cutoffs = get_ref<M>(*p.cutoff);
    ref.physics = get_ref<M>(*p.physics);
    ref.rng = get_ref<M>(*p.rng);
    ref.sim = get_ref<M>(*p.sim);
    ref.init = get_ref<M>(*p.init);

    CELER_ENSURE(ref);
    return ref;
}

//---------------------------------------------------------------------------//

ActionId find_along_step_id(ActionRegistry const& reg)
{
    for (auto aidx : range(reg.num_actions()))
    {
        // Get abstract action shared pointer and see if it's explicit
        auto const& base = reg.action(ActionId{aidx});
        if (auto expl
            = std::dynamic_pointer_cast<ExplicitActionInterface const>(base))
        {
            if (expl->order() == ActionOrder::along)
            {
                return expl->action_id();
            }
        }
    }
    return {};
}

//---------------------------------------------------------------------------//
class ImplicitGeometryAction final : public ConcreteAction
{
  public:
    // Construct with ID and label
    using ConcreteAction::ConcreteAction;
};

//---------------------------------------------------------------------------//
class ImplicitSimAction final : public ConcreteAction
{
  public:
    // Construct with ID and label
    using ConcreteAction::ConcreteAction;
};
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with all problem data, creating some actions too.
 */
CoreParams::CoreParams(Input input) : input_(std::move(input))
{
#define CP_VALIDATE_INPUT(MEMBER) \
    CELER_VALIDATE(input_.MEMBER, \
                   << "core input is missing " << #MEMBER << " data")
    CP_VALIDATE_INPUT(geometry);
    CP_VALIDATE_INPUT(material);
    CP_VALIDATE_INPUT(geomaterial);
    CP_VALIDATE_INPUT(particle);
    CP_VALIDATE_INPUT(cutoff);
    CP_VALIDATE_INPUT(physics);
    CP_VALIDATE_INPUT(rng);
    CP_VALIDATE_INPUT(sim);
    CP_VALIDATE_INPUT(init);
    CP_VALIDATE_INPUT(action_reg);
    CP_VALIDATE_INPUT(output_reg);
    CP_VALIDATE_INPUT(max_streams);
#undef CP_VALIDATE_INPUT

    CELER_EXPECT(input_);

    ScopedMem record_mem("CoreParams.construct");

    CoreScalars scalars;

    // Construct geometry action
    scalars.boundary_action = input_.action_reg->next_id();
    input_.action_reg->insert(
        std::make_shared<celeritas::generated::BoundaryAction>(
            scalars.boundary_action,
            "geo-boundary",
            "cross a geometry boundary"));

    // Construct implicit limit for propagator pausing midstep
    scalars.propagation_limit_action = input_.action_reg->next_id();
    input_.action_reg->insert(std::make_shared<ImplicitGeometryAction>(
        scalars.propagation_limit_action,
        "geo-propagation-limit",
        "pause due to propagation misbehavior"));

    // Construct action for killed looping tracks
    scalars.abandon_looping_action = input_.action_reg->next_id();
    input_.action_reg->insert(std::make_shared<ImplicitSimAction>(
        scalars.abandon_looping_action,
        "kill-looping",
        "kill due to too many field substeps"));

    // Define neutral and user-provided along-step actions
    std::shared_ptr<AlongStepNeutralAction const> along_step_neutral;
    scalars.along_step_user_action = find_along_step_id(*input_.action_reg);
    if (scalars.along_step_user_action)
    {
        // Test whether user-provided action is neutral
        along_step_neutral
            = std::dynamic_pointer_cast<AlongStepNeutralAction const>(
                input_.action_reg->action(scalars.along_step_user_action));
    }

    if (!along_step_neutral)
    {
        // Create neutral action if one doesn't exist
        along_step_neutral = std::make_shared<AlongStepNeutralAction>(
            input_.action_reg->next_id());
        input_.action_reg->insert(along_step_neutral);
    }
    scalars.along_step_neutral_action = along_step_neutral->action_id();
    if (!scalars.along_step_user_action)
    {
        // Use newly created neutral action by default
        CELER_LOG(warning) << "No along-step action specified: using neutral "
                              "particle propagation";
        scalars.along_step_user_action = scalars.along_step_neutral_action;
    }

    // Save maximum number of streams
    scalars.max_streams = input_.max_streams;

    // Construct initialize tracks action
    input_.action_reg->insert(std::make_shared<InitializeTracksAction>(
        input_.action_reg->next_id()));

    // TrackOrder doesn't have to be an argument right now and could be
    // captured but we're eventually expecting different TrackOrder for
    // different ActionOrder
    auto insert_sort_tracks_action = [this](const TrackOrder track_order) {
        input_.action_reg->insert(std::make_shared<SortTracksAction>(
            input_.action_reg->next_id(), track_order));
    };
    const TrackOrder track_order{input_.init->host_ref().track_order};
    switch (track_order)
    {
        case TrackOrder::partition_status:
        case TrackOrder::sort_step_limit_action:
        case TrackOrder::sort_along_step_action:
            // Sort with just the given track order
            insert_sort_tracks_action(track_order);
            break;
        case TrackOrder::sort_action:
            // Sort twice
            insert_sort_tracks_action(TrackOrder::sort_step_limit_action);
            insert_sort_tracks_action(TrackOrder::sort_along_step_action);
            break;
        default:
            break;
    }

    // Construct extend from secondaries action
    input_.action_reg->insert(std::make_shared<ExtendFromSecondariesAction>(
        input_.action_reg->next_id()));

    // Save host reference
    host_ref_ = build_params_refs<MemSpace::host>(input_, scalars);
    if (celeritas::device())
    {
        device_ref_ = build_params_refs<MemSpace::device>(input_, scalars);
        // Copy device ref to device global memory
        device_ref_vec_ = DeviceVector<DeviceRef>(1);
        device_ref_vec_.copy_to_device({&device_ref_, 1});
    }

#if CELERITAS_USE_JSON
    // Save system diagnostic information
    input_.output_reg->insert(OutputInterfaceAdapter<Device>::from_const_ref(
        OutputInterface::Category::system, "device", celeritas::device()));
    input_.output_reg->insert(
        OutputInterfaceAdapter<KernelRegistry>::from_const_ref(
            OutputInterface::Category::system,
            "kernels",
            celeritas::kernel_registry()));
    input_.output_reg->insert(OutputInterfaceAdapter<MemRegistry>::from_const_ref(
        OutputInterface::Category::system, "memory", celeritas::mem_registry()));
    input_.output_reg->insert(OutputInterfaceAdapter<Environment>::from_const_ref(
        OutputInterface::Category::system, "environ", celeritas::environment()));
#endif
    input_.output_reg->insert(std::make_shared<BuildOutput>());

    // Save core diagnostic information
    input_.output_reg->insert(
        std::make_shared<GeoParamsOutput>(input_.geometry));
    input_.output_reg->insert(
        std::make_shared<MaterialParamsOutput>(input_.material));
    input_.output_reg->insert(
        std::make_shared<ParticleParamsOutput>(input_.particle));
    input_.output_reg->insert(
        std::make_shared<PhysicsParamsOutput>(input_.physics));
    input_.output_reg->insert(
        std::make_shared<ActionRegistryOutput>(input_.action_reg));

    CELER_ENSURE(host_ref_);
    CELER_ENSURE(host_ref_.scalars.max_streams == this->max_streams());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
