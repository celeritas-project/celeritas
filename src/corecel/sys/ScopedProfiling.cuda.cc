//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedProfiling.cuda.cc
//---------------------------------------------------------------------------//

#include "ScopedProfiling.hh"

#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <nvtx3/nvToolsExt.h>

#include "corecel/io/Logger.hh"

#include "Device.hh"
#include "Environment.hh"

namespace celeritas
{
namespace
{

//---------------------------------------------------------------------------//
/*!
 * Global registry for strings used by NVTX.
 *
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
 * Retrieve the handle for a given message.
 *
 * Insert it if it doesn't already exist.
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
    auto [iter, inserted] = [&message] {
        std::unique_lock lock(mutex);
        return message_registry().insert({message, {}});
    }();
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

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Whether profiling is enabled.
 *
 * This is true only if the \c CELER_ENABLE_PROFILING environment variable is
 * set to a non-empty value.
 */
bool ScopedProfiling::enable_profiling()
{
    static bool const result = [] {
        if (!celeritas::getenv("CELER_ENABLE_PROFILING").empty())
        {
            if (!celeritas::device())
            {
                CELER_LOG(warning) << "Disabling profiling support "
                                      "since no device is available";
                return false;
            }
            CELER_LOG(info) << "Enabling profiling support since the "
                               "'CELER_ENABLE_PROFILING' "
                               "environment variable is present and non-empty";
            return true;
        }
        return false;
    }();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Activate nvtx profiling with options.
 */
void ScopedProfiling::activate_(Input const& input) noexcept
{
    nvtxEventAttributes_t attributes_ = make_attributes(input);
    int result = nvtxDomainRangePushEx(domain_handle(), &attributes_);
    if (result < 0)
    {
        activated_ = false;
        CELER_LOG(warning) << "Failed to activate profiling domain '"
                           << input.name << "'";
    }
}

//---------------------------------------------------------------------------//
/*!
 * End the profiling range.
 */
void ScopedProfiling::deactivate_() noexcept
{
    int result = nvtxDomainRangePop(domain_handle());
    if (result < 0)
    {
        CELER_LOG(warning) << "Failed to deactivate profiling domain";
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
