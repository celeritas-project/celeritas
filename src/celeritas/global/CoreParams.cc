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

#include "corecel/Assert.hh"
#include "corecel/data/Ref.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/geo/GeoMaterialParams.hh"  // IWYU pragma: keep
#include "celeritas/geo/GeoParams.hh"  // IWYU pragma: keep
#include "celeritas/geo/generated/BoundaryAction.hh"
#include "celeritas/mat/MaterialParams.hh"  // IWYU pragma: keep
#include "celeritas/phys/CutoffParams.hh"  // IWYU pragma: keep
#include "celeritas/phys/ParticleParams.hh"  // IWYU pragma: keep
#include "celeritas/phys/PhysicsParams.hh"  // IWYU pragma: keep
#include "celeritas/random/RngParams.hh"  // IWYU pragma: keep
#include "celeritas/track/ExtendFromSecondariesAction.hh"
#include "celeritas/track/InitializeTracksAction.hh"
#include "celeritas/track/SimParams.hh"  // IWYU pragma: keep
#include "celeritas/track/TrackInitParams.hh"  // IWYU pragma: keep

#include "ActionInterface.hh"
#include "ActionRegistry.hh"  // IWYU pragma: keep

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
class ImplicitGeometryAction final : public ImplicitActionInterface,
                                     public ConcreteAction
{
  public:
    // Construct with ID and label
    using ConcreteAction::ConcreteAction;
};

//---------------------------------------------------------------------------//
class ImplicitSimAction final : public ImplicitActionInterface,
                                public ConcreteAction
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
    CP_VALIDATE_INPUT(max_streams);
#undef CP_VALIDATE_INPUT

    CELER_EXPECT(input_);

    CoreScalars scalars;

    // Construct geometry action
    scalars.boundary_action = input_.action_reg->next_id();
    input_.action_reg->insert(
        std::make_shared<celeritas::generated::BoundaryAction>(
            scalars.boundary_action, "geo-boundary", "Boundary crossing"));

    // Construct implicit limit for propagator pausing midstep
    scalars.propagation_limit_action = input_.action_reg->next_id();
    input_.action_reg->insert(std::make_shared<ImplicitGeometryAction>(
        scalars.propagation_limit_action,
        "geo-propagation-limit",
        "Propagation substep/range limit"));

    // Construct action for killed looping tracks
    scalars_.abandon_looping_action = input_.action_reg->next_id();
    input_.action_reg->insert(
        std::make_shared<ImplicitSimAction>(scalars_.abandon_looping_action,
                                            "abandon-looping",
                                            "Abandoned looping track"));

    // Save maximum number of streams
    scalars.max_streams = input_.max_streams;

    // Construct initialize tracks action
    input_.action_reg->insert(std::make_shared<InitializeTracksAction>(
        input_.action_reg->next_id()));

    // Construct extend from secondaries action
    input_.action_reg->insert(std::make_shared<ExtendFromSecondariesAction>(
        input_.action_reg->next_id()));

    // Save host reference
    host_ref_ = build_params_refs<MemSpace::host>(input_, scalars);
    if (celeritas::device())
    {
        device_ref_ = build_params_refs<MemSpace::device>(input_, scalars);
    }
    CELER_ENSURE(host_ref_);
    CELER_ENSURE(host_ref_.scalars.max_streams == this->max_streams());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
