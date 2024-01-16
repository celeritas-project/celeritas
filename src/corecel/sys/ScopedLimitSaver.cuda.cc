//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedLimitSaver.cuda.cc
//---------------------------------------------------------------------------//
#include "ScopedLimitSaver.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Static data
Array<cudaLimit, 2> const ScopedLimitSaver::cuda_attrs_{
    {cudaLimitStackSize, cudaLimitMallocHeapSize}};
Array<char const*, 2> const ScopedLimitSaver::cuda_attr_labels_{
    {"stack size", "heap size"}};

//---------------------------------------------------------------------------//
/*!
 * Save attributes.
 */
ScopedLimitSaver::ScopedLimitSaver()
{
    for (auto i : range(orig_limits_.size()))
    {
        CELER_CUDA_CALL(cudaDeviceGetLimit(&orig_limits_[i], cuda_attrs_[i]));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Restore and possibly warn about changed attributes.
 */
ScopedLimitSaver::~ScopedLimitSaver()
{
    try
    {
        for (auto i : range(orig_limits_.size()))
        {
            std::size_t temp;
            CELER_CUDA_CALL(cudaDeviceGetLimit(&temp, cuda_attrs_[i]));
            if (temp != orig_limits_[i])
            {
                CELER_LOG(info)
                    << "CUDA " << cuda_attr_labels_[i] << " was changed from "
                    << orig_limits_[i] << " to " << temp
                    << "; restoring to original values";
                CELER_CUDA_CALL(
                    cudaDeviceSetLimit(cuda_attrs_[i], orig_limits_[i]));
            }
        }
    }
    catch (std::exception const& e)
    {
        CELER_LOG(error) << "Failed to restore CUDA device limits: "
                         << e.what();
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
