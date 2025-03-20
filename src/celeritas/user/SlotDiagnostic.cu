//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SlotDiagnostic.cu
//---------------------------------------------------------------------------//
#include "SlotDiagnostic.hh"

#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "detail/SlotDiagnosticExecutor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
void SlotDiagnostic::step(CoreParams const& params, CoreStateDevice& state) const
{
    // Allocate temporary device memory
    DeviceVector<int> device_buffer(state.size(), state.stream_id());

    TrackExecutor execute{
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::SlotDiagnosticExecutor{ObserverPtr{device_buffer.data()}}};

    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(*this, params, state, execute);

    // Copy IDs directly into host buffer
    device_buffer.copy_to_host(this->get_host_buffer(state.aux()));
    device_buffer = {};

    // Write IDs to the file
    this->write_buffer(state.aux());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
