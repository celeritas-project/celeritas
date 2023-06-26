//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedDeviceProfiling.cuda.cc
//---------------------------------------------------------------------------//
#include "ScopedDeviceProfiling.hh"

#include "celeritas_config.h"
#include "corecel/device_runtime_api.h"

// Profiler API isn't included with regular CUDA API headers
#include <nvtx3/nvToolsExt.h>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

#include "Device.hh"

namespace celeritas
{
namespace
{
nvtxDomainHandle_t domain_handle()
{
    static nvtxDomainHandle_t domain = nvtxDomainCreateA("celeritas");
    return domain;
}
nvtxEventAttributes_t make_attributes(std::string const& name)
{
    nvtxEventAttributes_t attributes;
    attributes.version = NVTX_VERSION;
    attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attributes.colorType = NVTX_COLOR_ARGB;
    attributes.color = 0xff00ff00;
    attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attributes.message.ascii = name.c_str();
    attributes.payloadType = NVTX_PAYLOAD_TYPE_INT32;
    attributes.payload.iValue = 0;
    attributes.category = 0;
    return attributes;
}
}  // namespace
//---------------------------------------------------------------------------//
/*!
 * Activate device profiling.
 */
ScopedDeviceProfiling::ScopedDeviceProfiling(std::string const& name)
{
    if (celeritas::device())
    {
        nvtxEventAttributes_t attributes_ = make_attributes(name);
        nvtxDomainRangePushEx(domain_handle(), &attributes_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Deactivate device profiling if this function activated it.
 */
ScopedDeviceProfiling::~ScopedDeviceProfiling()
{
    if (celeritas::device())
    {
        nvtxDomainRangePop(domain_handle());
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
