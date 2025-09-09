//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedLimitSaver.cuda.cc
//---------------------------------------------------------------------------//
#include "ScopedLimitSaver.hh"

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
Array<cudaLimit, 2> const g_attrs{
    {cudaLimitStackSize, cudaLimitMallocHeapSize}};
Array<char const*, 2> const g_labels{{"stack size", "heap size"}};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Save attributes.
 */
ScopedLimitSaver::ScopedLimitSaver()
{
    static_assert(g_attrs.size() == decltype(orig_limits_){}.size()
                  && g_labels.size() == g_attrs.size());
    for (auto i : range(orig_limits_.size()))
    {
        CELER_DEVICE_API_CALL(DeviceGetLimit(&orig_limits_[i], g_attrs[i]));
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
            CELER_DEVICE_API_CALL(DeviceGetLimit(&temp, g_attrs[i]));
            if (temp != orig_limits_[i])
            {
                CELER_LOG(info)
                    << "CUDA " << g_labels[i] << " was changed from "
                    << orig_limits_[i] << " to " << temp
                    << "; restoring to original values";
                CELER_DEVICE_API_CALL(
                    DeviceSetLimit(g_attrs[i], orig_limits_[i]));
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
