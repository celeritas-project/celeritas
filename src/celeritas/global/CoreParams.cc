//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreParams.cc
//---------------------------------------------------------------------------//
#include "CoreParams.hh"

#include <string>
#include <type_traits>
#include <utility>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/data/AuxParamsRegistry.hh"  // IWYU pragma: keep
#include "corecel/data/Ref.hh"
#include "corecel/io/BuildOutput.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputInterfaceAdapter.hh"
#include "corecel/io/OutputRegistry.hh"  // IWYU pragma: keep
#include "corecel/sys/ActionRegistry.hh"  // IWYU pragma: keep
#include "corecel/sys/ActionRegistryOutput.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/DeviceIO.json.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/EnvironmentIO.json.hh"
#include "corecel/sys/KernelRegistry.hh"
#include "corecel/sys/KernelRegistryIO.json.hh"
#include "corecel/sys/MemRegistry.hh"
#include "corecel/sys/MemRegistryIO.json.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "corecel/sys/ScopedMem.hh"
#include "geocel/GeoParamsOutput.hh"
#include "celeritas/alongstep/AlongStepNeutralAction.hh"
#include "celeritas/em/params/WentzelOKVIParams.hh"  // IWYU pragma: keep
#include "celeritas/geo/GeoMaterialParams.hh"  // IWYU pragma: keep
#include "celeritas/geo/GeoParams.hh"  // IWYU pragma: keep
#include "celeritas/geo/detail/BoundaryAction.hh"
#include "celeritas/mat/MaterialParams.hh"  // IWYU pragma: keep
#include "celeritas/mat/MaterialParamsOutput.hh"
#include "celeritas/phys/CutoffParams.hh"  // IWYU pragma: keep
#include "celeritas/phys/ParticleParams.hh"  // IWYU pragma: keep
#include "celeritas/phys/ParticleParamsOutput.hh"
#include "celeritas/phys/PhysicsParams.hh"  // IWYU pragma: keep
#include "celeritas/phys/PhysicsParamsOutput.hh"
#include "celeritas/phys/detail/TrackingCutAction.hh"
#include "celeritas/random/RngParams.hh"  // IWYU pragma: keep
#include "celeritas/track/ExtendFromPrimariesAction.hh"
#include "celeritas/track/ExtendFromSecondariesAction.hh"
#include "celeritas/track/InitializeTracksAction.hh"
#include "celeritas/track/SimParams.hh"  // IWYU pragma: keep
#include "celeritas/track/SortTracksAction.hh"
#include "celeritas/track/TrackInitParams.hh"  // IWYU pragma: keep

#include "ActionInterface.hh"

#include "detail/CoreSizes.json.hh"

#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
#    include "orange/OrangeParams.hh"  // IWYU pragma: keep
#    include "orange/OrangeParamsOutput.hh"
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
#    include "geocel/vg/VecgeomParams.hh"  // IWYU pragma: keep
#    include "geocel/vg/VecgeomParamsOutput.hh"
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
    if (p.wentzel)
    {
        ref.wentzel = get_ref<M>(*p.wentzel);
    }

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
            = std::dynamic_pointer_cast<CoreStepActionInterface const>(base))
        {
            if (expl->order() == StepActionOrder::along)
            {
                return expl->action_id();
            }
        }
    }
    return {};
}

//---------------------------------------------------------------------------//
class PropagationLimitAction final : public StaticConcreteAction
{
  public:
    //! Construct with ID
    explicit PropagationLimitAction(ActionId id)
        : StaticConcreteAction(id,
                               "geo-propagation-limit",
                               "pause due to propagation misbehavior")
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

    //// POST-STEP ACTIONS ////

    // Construct geometry boundary action
    scalars.boundary_action = reg->next_id();
    reg->insert(make_shared<celeritas::detail::BoundaryAction>(
        scalars.boundary_action));

    // Construct action for killed looping tracks/error geometry
    // NOTE: due to ordering by {start, ID}, TrackingCutAction *must*
    // be after BoundaryAction
    scalars.tracking_cut_action = reg->next_id();
    reg->insert(
        make_shared<detail::TrackingCutAction>(scalars.tracking_cut_action));

    //// END ACTIONS ////

    // Construct extend from secondaries action
    reg->insert(make_shared<ExtendFromSecondariesAction>(reg->next_id()));

    return scalars;
}

//---------------------------------------------------------------------------//
auto get_core_sizes(CoreParams const& cp)
{
    auto const& init = *cp.init();

    detail::CoreSizes result;
    result.processes = comm_world().size();
    result.streams = cp.max_streams();

    // NOTE: quantities are *per-process* quantities: integrated over streams,
    // but not processes
    result.initializers = result.streams * init.capacity();
    result.tracks = result.streams * cp.tracks_per_stream();
    // Number of secondaries is currently based on track size
    result.secondaries = static_cast<size_type>(
        cp.physics()->host_ref().scalars.secondary_stack_factor
        * result.tracks);
    // Event IDs are the same across all threads so this is *not* multiplied by
    // streams
    result.events = init.max_events();

    return result;
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

    if (!input_.aux_reg)
    {
        input_.aux_reg = std::make_shared<AuxParamsRegistry>();
    }
    if (!input_.mpi_comm)
    {
        // Create with a non-owning reference to the world communicator. This
        // is safe since the world comm is a static variable whose lifetime
        // should extend beyond anything that uses shared params.
        input_.mpi_comm.reset(&celeritas::comm_world(),
                              [](MpiCommunicator const*) {});
    }

    ScopedMem record_mem("CoreParams.construct");

    // Add track initializer generators (TODO: user does this externally)
    auto primaries = ExtendFromPrimariesAction::make_and_insert(*this);
    CELER_ASSERT(primaries);

    // Construct always-on actions and save their IDs
    CoreScalars scalars = build_actions(input_.action_reg.get());

    // Construct optional track-sorting actions
    auto insert_sort_tracks_action = [this](TrackOrder const track_order) {
        input_.action_reg->insert(std::make_shared<SortTracksAction>(
            input_.action_reg->next_id(), track_order));
    };
    switch (TrackOrder track_order = input_.init->track_order())
    {
        case TrackOrder::none:
        case TrackOrder::init_charge:
        case TrackOrder::reindex_shuffle:
            break;
        case TrackOrder::reindex_status:
        case TrackOrder::reindex_step_limit_action:
        case TrackOrder::reindex_along_step_action:
        case TrackOrder::reindex_particle_type:
            // Sort with just the given track order
            insert_sort_tracks_action(track_order);
            break;
        case TrackOrder::reindex_both_action:
            // Sort twice
            insert_sort_tracks_action(TrackOrder::reindex_step_limit_action);
            insert_sort_tracks_action(TrackOrder::reindex_along_step_action);
            break;
        case TrackOrder::size_:
            CELER_ASSERT_UNREACHABLE();
    }

    // Save maximum number of streams
    scalars.max_streams = input_.max_streams;

    // Save non-owning pointer to core params for host diagnostics
    scalars.host_core_params = ObserverPtr{this};

    // Save host reference
    host_ref_ = build_params_refs<MemSpace::host>(input_, scalars);
    if (celeritas::device())
    {
        device_ref_ = build_params_refs<MemSpace::device>(input_, scalars);
        // Copy device ref to device global memory
        device_ref_vec_ = DeviceVector<DeviceRef>(1);
        device_ref_vec_.copy_to_device({&device_ref_, 1});
    }

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
    input_.output_reg->insert(std::make_shared<BuildOutput>());
    input_.output_reg->insert(
        OutputInterfaceAdapter<detail::CoreSizes>::from_rvalue_ref(
            OutputInterface::Category::internal,
            "core-sizes",
            get_core_sizes(*this)));

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

    // TODO: add output from auxiliary params/data

    CELER_LOG(status) << "Celeritas core setup complete";

    CELER_ENSURE(host_ref_);
    CELER_ENSURE(host_ref_.scalars.max_streams == this->max_streams());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
