//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Material.test.cu
//---------------------------------------------------------------------------//
#include "Material.test.hh"

#include <thrust/device_vector.h>
#include "base/Range.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "physics/material/MaterialTrackView.hh"

using thrust::raw_pointer_cast;

namespace celeritas_test
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void m_test_kernel(unsigned int const                  size,
                              MTestInput::MaterialParamsRef const params,
                              MTestInput::MaterialStateRef const  states,
                              const MaterialTrackState* const     init,
                              real_type*                          temperatures,
                              real_type*                          rad_len,
                              real_type*                          tot_z)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= size)
        return;

    MaterialTrackView mat_track(params, states, tid);

    // Initialize state
    mat_track = init[tid.get()];
    CELER_ASSERT(mat_track.material_id() == init[tid.get()].material_id);

    // Get material properties
    const auto& mat         = mat_track.material_view();
    temperatures[tid.get()] = mat.temperature();
    rad_len[tid.get()]      = mat.radiation_length();

    // Fill elements with finctional cross sections
    celeritas::Span<real_type> scratch = mat_track.element_scratch();

    for (auto ec : celeritas::range(mat.num_elements()))
    {
        // Pretend to calculate cross section for the ec'th element
        const auto& element = mat.element_view(ElementComponentId{ec});
        scratch[ec]         = static_cast<real_type>(element.atomic_number());
    }

    real_type tz = 0.0;
    for (auto ec : celeritas::range(mat.num_elements()))
    {
        // Get its atomic number weighted by its fractional number density
        tz += scratch[ec] * mat.get_element_density(ElementComponentId{ec});
    }
    tot_z[tid.get()] = tz;
}
} // namespace

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

    static const celeritas::KernelParamCalculator calc_launch_params(
        m_test_kernel, "m_test");
    auto params = calc_launch_params(init.size());
    m_test_kernel<<<params.grid_size, params.block_size>>>(
        init.size(),
        input.params,
        input.states,
        raw_pointer_cast(init.data()),
        raw_pointer_cast(temperatures.data()),
        raw_pointer_cast(rad_len.data()),
        raw_pointer_cast(tot_z.data()));
    CELER_CUDA_CHECK_ERROR();
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
