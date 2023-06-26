//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedProfiling.nvtx.cc
//---------------------------------------------------------------------------//

#include "ScopedProfiling.hh"

#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <nvtx3/nvToolsExt.h>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace
{

//---------------------------------------------------------------------------//
/*!
 * Global registry for strings used by NVTX.
 * This Implemented as a free function instead of a static in
 * ScopedProfiling to hide the NVTX dependency from user of the
 * interface.
 */
std::unordered_map<std::string, nvtxStringHandle_t>& message_registry()
{
    static std::unordered_map<std::string, nvtxStringHandle_t> registry;
    return registry;
}

nvtxDomainHandle_t domain_handle()
{
    static nvtxDomainHandle_t domain = nvtxDomainCreateA("celeritas");
    return domain;
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve the handle for a given message, insert it if it doesn't already
 * exists
 */
nvtxStringHandle_t message_handle_for(std::string const& message)
{
    static std::shared_mutex mutex;

    {
        std::shared_lock lock(mutex);
        auto message_handle = message_registry().find(message);
        if (message_handle != message_registry().end())
        {
            return message_handle->second;
        }
    }
    // we did not find the handle, insert it
    std::unique_lock lock(mutex);
    auto message_handle = message_registry().find(message);
    // recheck that nobody inserted the same message as we had to release the
    // shared lock not strictly necessary as insert will do nothing if the key
    // already exists
    if (message_handle == message_registry().end())
    {
        auto handle
            = nvtxDomainRegisterStringA(domain_handle(), message.c_str());
        message_registry().insert({message, handle});
        return handle;
    }
    else
    {
        return message_handle->second;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Create EventAttribute with a specific name
 */
nvtxEventAttributes_t make_attributes(ScopedProfiling::Input const& input)
{
    nvtxEventAttributes_t attributes;
    attributes.version = NVTX_VERSION;
    attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attributes.colorType = NVTX_COLOR_ARGB;
    attributes.color = input.color;
    attributes.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    attributes.message.registered = message_handle_for(input.name);
    attributes.payloadType = NVTX_PAYLOAD_TYPE_INT32;
    attributes.payload.iValue = input.payload;
    attributes.category = input.category;
    return attributes;
}
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Activate nvtx profiling.
 */
ScopedProfiling::ScopedProfiling(Input const input)
{
    nvtxEventAttributes_t attributes_ = make_attributes(input);
    nvtxDomainRangePushEx(domain_handle(), &attributes_);
}

//---------------------------------------------------------------------------//
/*!
 * Deactivate device profiling if this function activated it.
 */
ScopedProfiling::~ScopedProfiling()
{
    nvtxDomainRangePop(domain_handle());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
