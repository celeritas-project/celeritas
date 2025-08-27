//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/CoreParams.cc
//---------------------------------------------------------------------------//
#include "CoreParams.hh"

#include "corecel/io/Logger.hh"
#include "corecel/random/params/RngParams.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/ScopedMem.hh"
#include "geocel/SurfaceParams.hh"
#include "celeritas/geo/CoreGeoParams.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/GeneratorRegistry.hh"
#include "celeritas/track/SimParams.hh"

#include "CoreState.hh"
#include "MaterialParams.hh"
#include "PhysicsParams.hh"
#include "action/AlongStepAction.hh"
#include "action/LocateVacanciesAction.hh"
#include "action/PreStepAction.hh"
#include "action/TrackingCutAction.hh"
#include "surface/SurfacePhysicsParams.hh"

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
    ref.physics = get_ref<M>(*p.physics);
    ref.surface = get_ref<M>(*p.surface);
    ref.surface_physics = get_ref<M>(*p.surface_physics);
    ref.rng = get_ref<M>(*p.rng);

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

    //// PRE-STEP ACTIONS ////

    reg->insert(make_shared<PreStepAction>(reg->next_id()));

    //// ALONG-STEP ACTIONS ////

    reg->insert(make_shared<AlongStepAction>(reg->next_id()));

    //// POST-STEP ACTIONS ////

    // TODO: process selection action (or constructed by physics?)

    scalars.tracking_cut_action = reg->next_id();
    reg->insert(make_shared<TrackingCutAction>(scalars.tracking_cut_action));

    //// END ACTIONS ////

    reg->insert(make_shared<LocateVacanciesAction>(reg->next_id()));

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
    CP_VALIDATE_INPUT(physics);
    CP_VALIDATE_INPUT(rng);
    CP_VALIDATE_INPUT(surface);
    CP_VALIDATE_INPUT(surface_physics);
    CP_VALIDATE_INPUT(action_reg);
    CP_VALIDATE_INPUT(gen_reg);
    CP_VALIDATE_INPUT(max_streams);
#undef CP_VALIDATE_INPUT

    CELER_EXPECT(input_);

    // Build detector params based on input detector labels vector. If label
    // returns false, create an empty label vector.
    if (input_.detector_labels)
    {
        detectors_ = std::make_shared<SDParams>(*(input_.detector_labels),
                                                *(input_.geometry));
    }
    else
    {
        detectors_ = std::make_shared<SDParams>();
    }

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
