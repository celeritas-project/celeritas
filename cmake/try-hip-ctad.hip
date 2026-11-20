//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file try-hip-ctad.hip
//! Check whether unannotated deduction guides can be used in device code.
//---------------------------------------------------------------------------//

template<class T>
struct Wrapper
{
    __host__ __device__ explicit Wrapper(T v) : value{v} {}

    T value;
};

// Unannotated deduction guide: older HIP compilers treat it as host-only, and
// newer ones reject host/device annotations on it
template<class T>
Wrapper(T) -> Wrapper<T>;

__global__ void use_ctad(int* result)
{
    Wrapper w{1};
    *result = w.value;
}
