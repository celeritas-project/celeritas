//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoRun.cc
//---------------------------------------------------------------------------//
#include "LDemoRun.hh"

#include "base/CollectionStateStore.hh"
#include "comm/Logger.hh"
#include "LDemoParams.hh"
#include "LDemoInterface.hh"
#include "LDemoKernel.hh"

using namespace celeritas;

namespace demo_loop
{
namespace
{
//---------------------------------------------------------------------------//
template<class P, MemSpace M>
struct ParamsGetter;

template<class P>
struct ParamsGetter<P, MemSpace::host>
{
    const P& params_;

    auto operator()() const -> decltype(auto)
    {
        return params_.host_pointers();
    }
};

template<class P>
struct ParamsGetter<P, MemSpace::device>
{
    const P& params_;

    auto operator()() const -> decltype(auto)
    {
        return params_.device_pointers();
    }
};

template<MemSpace M, class P>
decltype(auto) get_pointers(const P& params)
{
    return ParamsGetter<P, M>{params}();
}

//---------------------------------------------------------------------------//
template<MemSpace M>
ParamsData<Ownership::const_reference, M>
build_params_refs(const LDemoParams& p)
{
    ParamsData<Ownership::const_reference, M> ref;
    ref.geometry  = get_pointers<M>(*p.geometry);
    ref.materials = get_pointers<M>(*p.materials);
    CELER_NOT_IMPLEMENTED("TODO: add remaining references");

    return ref;
}

//---------------------------------------------------------------------------//
/*!
 * Launch interaction kernels for all applicable models.
 *
 * For now, just launch *all* the models.
 */
void launch_models()
{
    CELER_NOT_IMPLEMENTED("TODO: add remaining processes");
    // Create ModelInteractPointers
    // Loop over physics models IDs
    // Invoke `interact`
    // TODO: for this to work on host, we'll need to template
    // ModelInterface on MemSpace and overload the `interact`
    // method on Model to work with device pointers.
}

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
LDemoResult run_gpu(LDemoArgs args)
{
    CELER_EXPECT(args);

    // Load all the problem data
    LDemoParams params = load_params(args);

    // Create param interfaces (TODO unify with sim/TrackInterface )
    ParamsDeviceRef params_ref = build_params_refs<MemSpace::device>(params);

    // Create states (TODO state store?)
    StateData<Ownership::value, MemSpace::device> state_storage;
    resize(&state_storage,
           build_params_refs<MemSpace::host>(params),
           args.num_tracks);
    StateDeviceRef states_ref = make_ref(state_storage);

    CELER_NOT_IMPLEMENTED("TODO: stepping loop");

    // - Initialize fixed number of primaries (isotropic samples?)

    bool any_alive = true;
    while (any_alive)
    {
        demo_loop::pre_step(params_ref, states_ref);
        // - Geometry propagate
        // - Deposit energy/slow down
        // - Select model
        launch_models();
        // - Process interactions and secondaries:
        //  - Increment energy deposition from interaction
        //  - Kill secondaries below production cuts and deposit their energy
        //  too
        // - Create primaries from secondaries
    }
}

//---------------------------------------------------------------------------//
} // namespace demo_loop
