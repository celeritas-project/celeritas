//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/CoreParams.cc
//---------------------------------------------------------------------------//
#include "CoreParams.hh"

#include "corecel/io/Logger.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/ScopedMem.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/random/RngParams.hh"
#include "celeritas/track/SimParams.hh"
#include "celeritas/track/TrackInitParams.hh"

#include "CoreState.hh"
#include "MaterialParams.hh"
#include "TrackInitParams.hh"
#include "action/BoundaryAction.hh"

namespace celeritas
{
namespace optical
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
    ref.material = get_ref<M>(*p.material);
    // TODO: ref.physics = get_ref<M>(*p.physics);
    ref.rng = get_ref<M>(*p.rng);
    ref.sim = get_ref<M>(*p.sim);
    ref.init = get_ref<M>(*p.init);

    CELER_ENSURE(ref);
    return ref;
}

//---------------------------------------------------------------------------//
/*!
 * Construct always-required actions and set IDs.
 */
CoreScalars build_actions(ActionRegistry* reg)
{
    using std::make_shared;

    CoreScalars scalars;

    //// START ACTIONS ////

#if 0
    reg->insert(make_shared<InitializeTracksAction>(reg->next_id()));
#endif

    //// PRE-STEP ACTIONS ////

    //// POST-STEP ACTIONS ////

    // Construct geometry boundary action
    // TODO: it might make more sense to build these actions right before
    // making the action group: re-examine once we add a surface physics
    // manager
    scalars.boundary_action = reg->next_id();
    reg->insert(make_shared<BoundaryAction>(scalars.boundary_action));

    //// END ACTIONS ////

    // TODO: extend from secondaries action

    return scalars;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with all problem data, creating some actions too.
 */
CoreParams::CoreParams(Input&& input) : input_(std::move(input))
{
#define CP_VALIDATE_INPUT(MEMBER) \
    CELER_VALIDATE(input_.MEMBER, \
                   << "optical core input is missing " << #MEMBER << " data")
    CP_VALIDATE_INPUT(geometry);
    CP_VALIDATE_INPUT(material);
    // TODO: CP_VALIDATE_INPUT(physics);
    CP_VALIDATE_INPUT(rng);
    CP_VALIDATE_INPUT(sim);
    CP_VALIDATE_INPUT(init);
    CP_VALIDATE_INPUT(action_reg);
    CP_VALIDATE_INPUT(max_streams);
#undef CP_VALIDATE_INPUT

    CELER_EXPECT(input_);

    ScopedMem record_mem("optical::CoreParams.construct");

    // Construct always-on actions and save their IDs
    CoreScalars scalars = build_actions(input_.action_reg.get());

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

    CELER_LOG(status) << "Celeritas optical setup complete";

    CELER_ENSURE(host_ref_);
    CELER_ENSURE(host_ref_.scalars.max_streams == this->max_streams());
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
