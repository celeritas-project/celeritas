//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedProfiling.cuda.cc
//! \brief The nvtx implementation of \c ScopedProfiling
//---------------------------------------------------------------------------//
#include "ScopedProfiling.hh"

#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <nvtx3/nvToolsExt.h>

#include "corecel/io/Logger.hh"

#include "Environment.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Library-wide handle to the domain name.
 */
nvtxDomainHandle_t domain_handle()
{
    static nvtxDomainHandle_t const domain = nvtxDomainCreateA("celeritas");
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
    static std::unordered_map<std::string, nvtxStringHandle_t> registry;
    static std::shared_mutex mutex;

    {
        std::shared_lock lock(mutex);

        if (auto message_handle = registry.find(message);
            message_handle != registry.end())
        {
            return message_handle->second;
        }
    }

    // We did not find the handle; try to insert it
    auto [iter, inserted] = [&message] {
        std::unique_lock lock(mutex);
        return registry.insert({message, {}});
    }();
    if (inserted)
    {
        // Register the domain
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
    nvtxEventAttributes_t attributes = {};  // Initialize all attributes to
                                            // zero
    attributes.version = NVTX_VERSION;
    attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    if (input.color)
    {
        attributes.colorType = NVTX_COLOR_ARGB;
        attributes.color = input.color;
    }
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
bool ScopedProfiling::use_profiling()
{
    static bool const result = [] {
        if (!celeritas::getenv("CELER_ENABLE_PROFILING").empty())
        {
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
 *
 * The call to NVTX is checked for validity (it should return a nonnegative
 * number) except that we ignore -1 because that seems to be returned even when
 * the call produces correct ranges in the profiling output.
 */
void ScopedProfiling::activate(Input const& input) noexcept
{
    nvtxEventAttributes_t attributes = make_attributes(input);
    int depth = nvtxDomainRangePushEx(domain_handle(), &attributes);
    if (depth < -1)
    {
        activated_ = false;

        // Warn about failures, but only twice
        constexpr int max_warnings{2};
        static int num_warnings{0};
        if (num_warnings < max_warnings)
        {
            ++num_warnings;

            CELER_LOG(warning)
                << "Failed to activate profiling domain '" << input.name
                << "' (error code " << depth << ")";
            if (num_warnings == 1)
            {
                CELER_LOG(info) << "Perhaps you're not running through `nsys` "
                                   "or using the `celeritas` domain?";
            }

            if (num_warnings == max_warnings)
            {
                CELER_LOG(info) << "Suppressing future scoped profiling "
                                   "warnings";
            }
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * End the profiling range.
 */
void ScopedProfiling::deactivate() noexcept
{
    int result = nvtxDomainRangePop(domain_handle());
    if (result < -1)
    {
        CELER_LOG(warning)
            << "Failed to deactivate profiling domain (error code " << result
            << ")";
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
