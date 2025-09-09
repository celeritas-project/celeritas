//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mat/Material.test.cu
//---------------------------------------------------------------------------//
#include "Material.test.hh"

#include <thrust/device_vector.h>

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/cont/Range.hh"
#include "corecel/math/Quantity.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/mat/MaterialTrackView.hh"

using thrust::raw_pointer_cast;

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void m_test_kernel(unsigned int const size,
                              MTestInput::MaterialParamsRef const params,
                              MTestInput::MaterialStateRef const states,
                              MaterialTrackState const* const init,
                              real_type* temperatures,
                              real_type* rad_len,
                              real_type* tot_z)
{
    auto tid = TrackSlotId{KernelParamCalculator::thread_id().unchecked_get()};
    if (tid.get() >= size)
        return;

    MaterialTrackView mat_track(params, states, tid);

    // Initialize state
    mat_track = init[tid.get()];
    CELER_ASSERT(mat_track.material_id() == init[tid.get()].material_id);

    // Get material properties
    auto const& mat = mat_track.material_record();
    temperatures[tid.get()] = mat.temperature();
    rad_len[tid.get()]
        = native_value_to<units::CmLength>(mat.radiation_length()).value();

    // Fill elements with functional cross sections
    Span<real_type> scratch = mat_track.element_scratch();

    for (auto ec : range(mat.num_elements()))
    {
        // Pretend to calculate cross section for the ec'th element
        auto const& element = mat.element_record(ElementComponentId{ec});
        scratch[ec]
            = static_cast<real_type>(element.atomic_number().unchecked_get());
    }

    real_type tz = 0.0;
    for (auto ec : range(mat.num_elements()))
    {
        // Get its atomic number weighted by its fractional number density
        tz += scratch[ec]
              * native_value_to<units::InvCcDensity>(
                    mat.get_element_density(ElementComponentId{ec}))
                    .value();
    }
    tot_z[tid.get()] = tz;
}
}  // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
MTestOutput m_test(MTestInput const& input)
{
    thrust::device_vector<MaterialTrackState> init = input.init;
    thrust::device_vector<real_type> temperatures(input.size());
    thrust::device_vector<real_type> rad_len(input.size());
    thrust::device_vector<real_type> tot_z(input.size());

    CELER_LAUNCH_KERNEL(m_test,
                        init.size(),
                        0,
                        init.size(),
                        input.params,
                        input.states,
                        raw_pointer_cast(init.data()),
                        raw_pointer_cast(temperatures.data()),
                        raw_pointer_cast(rad_len.data()),
                        raw_pointer_cast(tot_z.data()));
    CELER_DEVICE_API_CALL(DeviceSynchronize());

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
}  // namespace test
}  // namespace celeritas
