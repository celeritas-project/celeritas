//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreParams.cc
//---------------------------------------------------------------------------//
#include "CoreParams.hh"

#include "corecel/Assert.hh"
#include "corecel/data/Ref.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/geo/generated/BoundaryAction.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/ActionManager.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/random/RngParams.hh"

#include "ActionManager.hh"

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
build_params_refs(const CoreParams::Input& p, CoreScalars scalars)
{
    CELER_EXPECT(scalars);

    CoreParamsData<Ownership::const_reference, M> ref;

    ref.scalars   = scalars;
    ref.geometry  = get_ref<M>(*p.geometry);
    ref.geo_mats  = get_ref<M>(*p.geomaterial);
    ref.materials = get_ref<M>(*p.material);
    ref.particles = get_ref<M>(*p.particle);
    ref.cutoffs   = get_ref<M>(*p.cutoff);
    ref.physics   = get_ref<M>(*p.physics);
    ref.rng       = get_ref<M>(*p.rng);

    CELER_ENSURE(ref);
    return ref;
}
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with all problem data, creating some actions too.
 */
CoreParams::CoreParams(Input input) : input_(std::move(input))
{
    CELER_EXPECT(input_);
    // Construct geometry action
    scalars_.boundary_action = input_.action_mgr->next_id();
    input_.action_mgr->insert(
        std::make_shared<celeritas::generated::BoundaryAction>(
            scalars_.boundary_action, "geo-boundary", "Boundary crossing"));

    // Save host reference
    host_ref_ = build_params_refs<MemSpace::host>(input_, scalars_);
    if (celeritas::device())
    {
        device_ref_ = build_params_refs<MemSpace::device>(input_, scalars_);
    }
    CELER_ENSURE(host_ref_);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
