//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/detail/NvtxUtils.cc
//! \brief Internal utilities for Nvtx implementation
// //---------------------------------------------------------------------------//
#include "NvtxUtils.hh"

#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace celeritas
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
nvtxStringHandle_t message_handle_for(std::string_view message)
{
    static std::unordered_map<std::string, nvtxStringHandle_t> registry;
    static std::shared_mutex mutex;

    std::string temp_message{message};
    {
        std::shared_lock lock(mutex);

        if (auto message_handle = registry.find(temp_message);
            message_handle != registry.end())
        {
            return message_handle->second;
        }
    }

    // We did not find the handle; try to insert it
    auto [iter, inserted] = [&temp_message] {
        std::unique_lock lock(mutex);
        return registry.insert({std::move(temp_message), {}});
    }();
    if (inserted)
    {
        // Register the domain
        iter->second
            = nvtxDomainRegisterStringA(domain_handle(), iter->first.c_str());
    }
    return iter->second;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
