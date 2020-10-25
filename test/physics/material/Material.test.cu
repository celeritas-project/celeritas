//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Material.test.cu
//---------------------------------------------------------------------------//
#include "Material.test.hh"

#include <thrust/device_vector.h>
#include "base/KernelParamCalculator.cuda.hh"
#include "physics/material/MaterialTrackView.hh"

using thrust::raw_pointer_cast;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void m_test_kernel(unsigned int const           size,
                              MaterialParamsPointers const params,
                              MaterialStatePointers const  states,
                              const MaterialTrackState*    init,
                              real_type*                   temperatures,
                              real_type*                   rad_len,
                              real_type*                   tot_z)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= size)
        return;

    MaterialTrackView mat_track(params, states, tid);

    // Initialize state
    mat_track = init[tid.get()];
    CHECK(mat_track.def_id() == init[tid.get()].def_id);

    // Get material properties
    const auto& mat         = mat_track.material_view();
    temperatures[tid.get()] = mat.temperature();
    rad_len[tid.get()]      = mat.radiation_length();

    // Loop over elements (TODO replace with range)
    real_type tz = 0.0;
    for (unsigned int ec = 0, ecmax = mat.num_elements(); ec != ecmax; ++ec)
    {
        // Get ec'th component of the current material
        const auto& element = mat.element_view(ElementComponentId{ec});

        // Get its atomic number weighted by its fractional number density
        tz += element.atomic_number()
              * mat.get_element_density(ElementComponentId{ec});
    }
    tot_z[tid.get()] = tz;
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
MTestOutput m_test(const MTestInput& input)
{
    thrust::device_vector<MaterialTrackState> init = input.init;
    thrust::device_vector<real_type>          temperatures(input.size());
    thrust::device_vector<real_type>          rad_len(input.size());
    thrust::device_vector<real_type>          tot_z(input.size());

    celeritas::KernelParamCalculator calc_launch_params;
    auto                             params = calc_launch_params(init.size());
    m_test_kernel<<<params.grid_size, params.block_size>>>(
        init.size(),
        input.params,
        input.states,
        raw_pointer_cast(init.data()),
        raw_pointer_cast(temperatures.data()),
        raw_pointer_cast(rad_len.data()),
        raw_pointer_cast(tot_z.data()));
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    MTestOutput result;
    result.temperatures.resize(init.size());
    result.rad_len.resize(init.size());
    result.tot_z.resize(init.size());

    thrust::copy(
        temperatures.begin(), temperatures.end(), result.temperatures.begin());
    thrust::copy(rad_len.begin(), rad_len.end(), result.rad_len.begin());
    thrust::copy(tot_z.begin(), tot_z.end(), result.tot_z.begin());
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
