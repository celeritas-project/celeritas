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

/**
 * @file
 *
 * The nvtx implementation of \c ScopedProfiling only does something when the
 * application using Celeritas is ran through a tool that supports nvtx, e.g.
 * nsight compute with the --nvtx argument. If this is not the case, API
 * calls to nvtx are disabled, doing noop.
 */

namespace celeritas
{
namespace
{

//---------------------------------------------------------------------------//
/*!
 * Global registry for strings used by NVTX.
 * This is implemented as a free function instead of a class static member in
 * ScopedProfiling to hide the NVTX dependency from users of the
 * interface.
 */
std::unordered_map<std::string, nvtxStringHandle_t>& message_registry()
{
    static std::unordered_map<std::string, nvtxStringHandle_t> registry;
    return registry;
}

//---------------------------------------------------------------------------//
/*!
 * Library-wide handle to the domain name.
 */
nvtxDomainHandle_t domain_handle()
{
    static nvtxDomainHandle_t domain = nvtxDomainCreateA("celeritas");
    return domain;
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve the handle for a given message. Insert it if it doesn't already
 * exists.
 */
nvtxStringHandle_t message_handle_for(std::string const& message)
{
    static std::shared_mutex mutex;

    {
        std::shared_lock lock(mutex);

        if (auto message_handle = message_registry().find(message);
            message_handle != message_registry().end())
        {
            return message_handle->second;
        }
    }
    // We did not find the handle; try to insert it
    std::unique_lock lock(mutex);
    auto [iter, inserted] = message_registry().insert({message, {}});
    if (inserted)
    {
        iter->second
            = nvtxDomainRegisterStringA(domain_handle(), message.c_str());
    }
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Create EventAttribute with a specific name.
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
 * Activate nvtx profiling with options.
 */
ScopedProfiling::ScopedProfiling(Input input)
{
    nvtxEventAttributes_t attributes_ = make_attributes(input);
    nvtxDomainRangePushEx(domain_handle(), &attributes_);
}

//---------------------------------------------------------------------------//
/*!
 * Activate nvtx profiling.
 */
ScopedProfiling::ScopedProfiling(std::string const& name)
    : ScopedProfiling{Input{name}}
{
}

//---------------------------------------------------------------------------//
/*!
 * End the profiling range.
 */
ScopedProfiling::~ScopedProfiling()
{
    nvtxDomainRangePop(domain_handle());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
