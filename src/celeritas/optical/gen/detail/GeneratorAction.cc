//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/GeneratorAction.cc
//---------------------------------------------------------------------------//
#include "GeneratorAction.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/CoreTrackData.hh"
#include "celeritas/optical/MaterialParams.hh"

#include "GeneratorExecutor.hh"
#include "OpticalGenAlgorithms.hh"
#include "../CherenkovData.hh"
#include "../CherenkovGenerator.hh"
#include "../ScintillationData.hh"
#include "../ScintillationGenerator.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID, data IDs, and optical properties.
 */
template<template<Ownership, MemSpace> class D, class G>
GeneratorAction<D, G>::GeneratorAction(Input&& inp)
    : action_id_(inp.action)
    , aux_id_{inp.aux}
    , optical_id_{inp.optical}
    , material_(inp.material)
    , shared_(inp.shared)
    , auto_flush_(inp.auto_flush)
    , buffer_capacity_(inp.buffer_capacity)
    , label_(inp.label)
{
    CELER_EXPECT(inp);
}

//---------------------------------------------------------------------------//
/*!
 * Build state data for a stream.
 */
template<template<Ownership, MemSpace> class D, class G>
auto GeneratorAction<D, G>::create_state(MemSpace m,
                                         StreamId id,
                                         size_type) const -> UPState
{
    if (m == MemSpace::host)
    {
        using StoreT = CollectionStateStore<GeneratorStateData, MemSpace::host>;
        auto result = std::make_unique<GeneratorState<MemSpace::host>>();
        result->store = StoreT{shared_->host_ref(), id, buffer_capacity_};
        CELER_ENSURE(*result);
        return result;
    }
    else if (m == MemSpace::device)
    {
        using StoreT
            = CollectionStateStore<GeneratorStateData, MemSpace::device>;
        auto result = std::make_unique<GeneratorState<MemSpace::device>>();
        result->store = StoreT{shared_->host_ref(), id, buffer_capacity_};
        CELER_ENSURE(*result);
        return result;
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Descriptive name of the action.
 */
template<template<Ownership, MemSpace> class D, class G>
std::string_view GeneratorAction<D, G>::description() const
{
    return "generate photons from optical distribution data";
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
template<template<Ownership, MemSpace> class D, class G>
void GeneratorAction<D, G>::step(CoreParams const& params,
                                 CoreStateHost& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
template<template<Ownership, MemSpace> class D, class G>
void GeneratorAction<D, G>::step(CoreParams const& params,
                                 CoreStateDevice& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical track initializers from distribution data.
 */
template<template<Ownership, MemSpace> class D, class G>
template<MemSpace M>
void GeneratorAction<D, G>::step_impl(CoreParams const& core_params,
                                      CoreState<M>& core_state) const
{
    auto& aux_state = get<GeneratorState<M>>(core_state.aux(), aux_id_);
    auto& optical_state
        = get<optical::CoreState<M>>(core_state.aux(), optical_id_);

    auto& photons = optical_state.counters().num_initializers;
    auto& num_new_photons = optical_state.counters().num_pending;

    if (photons + num_new_photons < auto_flush_)
    {
        // Below the threshold for launching the optical loop
        return;
    }

    auto initializers_size = optical_state.ref().init.initializers.size();
    CELER_VALIDATE(photons + num_new_photons <= initializers_size,
                   << "insufficient capacity (" << initializers_size
                   << ") for optical photon initializers (total capacity "
                      "requirement of "
                   << photons + num_new_photons << " and current size "
                   << photons << ")");

    auto& offload = aux_state.store.ref();
    auto& buffer_size = aux_state.buffer_size;
    if (buffer_size == 0)
    {
        // No new photons
        return;
    }

    // Calculate the cumulative sum of the number of photons in the buffered
    // distributions. These values are used to determine which thread will
    // generate initializers from which distribution
    auto count = inclusive_scan_photons(offload.distributions,
                                        offload.offsets,
                                        buffer_size,
                                        core_state.stream_id());
    optical_state.counters().num_generated += count;

    // Generate the optical photon initializers from the distribution data
    this->generate(core_params, core_state);

    // Update cumulative statistics
    aux_state.accum.distributions += buffer_size;
    aux_state.accum.photons += count;

    photons += count;
    num_new_photons -= count;
    buffer_size = 0;
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to generate optical photon initializers.
 */
template<template<Ownership, MemSpace> class D, class G>
void GeneratorAction<D, G>::generate(CoreParams const& core_params,
                                     CoreStateHost& core_state) const
{
    auto& aux_state
        = get<GeneratorState<MemSpace::native>>(core_state.aux(), aux_id_);
    auto& optical_state = get<optical::CoreState<MemSpace::native>>(
        core_state.aux(), optical_id_);

    TrackExecutor execute{
        core_params.ptr<MemSpace::native>(),
        core_state.ptr(),
        detail::GeneratorExecutor<D, G>{core_state.ptr(),
                                        material_->host_ref(),
                                        shared_->host_ref(),
                                        aux_state.store.ref(),
                                        optical_state.ptr(),
                                        aux_state.buffer_size,
                                        optical_state.counters()}};
    launch_action(*this, core_params, core_state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
template<template<Ownership, MemSpace> class D, class G>
void GeneratorAction<D, G>::generate(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class GeneratorAction<CherenkovData, CherenkovGenerator>;
template class GeneratorAction<ScintillationData, ScintillationGenerator>;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
