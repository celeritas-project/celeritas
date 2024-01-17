//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
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
#include "celeritas/geo/detail/BoundaryAction.hh"
#include "celeritas/global/ActionRegistryOutput.hh"
#include "celeritas/mat/MaterialParams.hh"  // IWYU pragma: keep
#include "celeritas/mat/MaterialParamsOutput.hh"
#include "celeritas/phys/CutoffParams.hh"  // IWYU pragma: keep
#include "celeritas/phys/ParticleParams.hh"  // IWYU pragma: keep
#include "celeritas/phys/ParticleParamsOutput.hh"
#include "celeritas/phys/PhysicsParams.hh"  // IWYU pragma: keep
#include "celeritas/phys/PhysicsParamsOutput.hh"
#include "celeritas/random/RngParams.hh"  // IWYU pragma: keep
#include "celeritas/track/ExtendFromPrimariesAction.hh"
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

#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
#    include "orange/OrangeParams.hh"
#    include "orange/OrangeParamsOutput.hh"
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
#    include "celeritas/ext/VecgeomParams.hh"
#    include "celeritas/ext/VecgeomParamsOutput.hh"
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
class PropagationLimitAction final : public ConcreteAction
{
  public:
    //! Construct with ID
    explicit PropagationLimitAction(ActionId id)
        : ConcreteAction(
            id, "geo-propagation-limit", "pause due to propagation misbehavior")
    {
    }
};

//---------------------------------------------------------------------------//
class AbandonLoopingAction final : public ConcreteAction
{
  public:
    //! Construct with ID
    explicit AbandonLoopingAction(ActionId id)
        : ConcreteAction(
            id, "kill-looping", "kill due to too many field substeps")
    {
    }
};

//---------------------------------------------------------------------------//
/*!
 * Construct always-required actions and set IDs.
 */
CoreScalars build_actions(ActionRegistry* reg)
{
    using std::make_shared;

    CoreScalars scalars;

    //// START ACTIONS ////

    // NOTE: due to ordering by {start, ID}, ExtendFromPrimariesAction *must*
    // precede InitializeTracksAction
    reg->insert(make_shared<ExtendFromPrimariesAction>(reg->next_id()));
    reg->insert(make_shared<InitializeTracksAction>(reg->next_id()));

    //// PRE-STEP ACTIONS ////

    //// ALONG-STEP ACTIONS ////

    // Define neutral and user-provided along-step actions
    std::shared_ptr<AlongStepNeutralAction const> along_step_neutral;
    scalars.along_step_user_action = find_along_step_id(*reg);
    if (scalars.along_step_user_action)
    {
        // Test whether user-provided action is neutral
        along_step_neutral
            = std::dynamic_pointer_cast<AlongStepNeutralAction const>(
                reg->action(scalars.along_step_user_action));
    }

    if (!along_step_neutral)
    {
        // Create neutral action if one doesn't exist
        along_step_neutral
            = make_shared<AlongStepNeutralAction>(reg->next_id());
        reg->insert(along_step_neutral);
    }
    scalars.along_step_neutral_action = along_step_neutral->action_id();
    if (!scalars.along_step_user_action)
    {
        // Use newly created neutral action by default
        CELER_LOG(warning) << "No along-step action specified: using neutral "
                              "particle propagation";
        scalars.along_step_user_action = scalars.along_step_neutral_action;
    }

    //// ALONG-STEP ACTIONS ////

    // Construct implicit limit for propagator pausing midstep
    scalars.propagation_limit_action = reg->next_id();
    reg->insert(
        make_shared<PropagationLimitAction>(scalars.propagation_limit_action));

    // Construct action for killed looping tracks
    scalars.abandon_looping_action = reg->next_id();
    reg->insert(
        make_shared<AbandonLoopingAction>(scalars.abandon_looping_action));

    //// POST-STEP ACTIONS ////

    // Construct geometry action
    scalars.boundary_action = reg->next_id();
    reg->insert(make_shared<celeritas::detail::BoundaryAction>(
        scalars.boundary_action));

    //// END ACTIONS ////

    // Construct extend from secondaries action
    reg->insert(make_shared<ExtendFromSecondariesAction>(reg->next_id()));

    return scalars;
}

//---------------------------------------------------------------------------//
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

    // Construct always-on actions and save their IDs
    CoreScalars scalars = build_actions(input_.action_reg.get());

    // Construct optional track-sorting actions
    auto insert_sort_tracks_action = [this](TrackOrder const track_order) {
        input_.action_reg->insert(std::make_shared<SortTracksAction>(
            input_.action_reg->next_id(), track_order));
    };
    switch (TrackOrder track_order = input_.init->host_ref().track_order)
    {
        case TrackOrder::partition_status:
        case TrackOrder::sort_step_limit_action:
        case TrackOrder::sort_along_step_action:
        case TrackOrder::sort_particle_type:
            // Sort with just the given track order
            insert_sort_tracks_action(track_order);
            break;
        case TrackOrder::sort_action:
            // Sort twice
            insert_sort_tracks_action(TrackOrder::sort_step_limit_action);
            insert_sort_tracks_action(TrackOrder::sort_along_step_action);
            break;
        case TrackOrder::unsorted:
        case TrackOrder::shuffled:
        case TrackOrder::size_:
            break;
    }

    // Save maximum number of streams
    scalars.max_streams = input_.max_streams;

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

#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
    input_.output_reg->insert(
        std::make_shared<OrangeParamsOutput>(input_.geometry));
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
    input_.output_reg->insert(
        std::make_shared<VecgeomParamsOutput>(input_.geometry));
#endif

    CELER_LOG(status) << "Celeritas core setup complete";

    CELER_ENSURE(host_ref_);
    CELER_ENSURE(host_ref_.scalars.max_streams == this->max_streams());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
