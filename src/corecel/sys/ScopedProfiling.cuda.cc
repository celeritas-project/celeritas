//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedProfiling.cuda.cc
//! \brief The nvtx implementation of \c ScopedProfiling
//---------------------------------------------------------------------------//
#include "ScopedProfiling.hh"

#include <nvtx3/nvToolsExt.h>

#include "corecel/io/Logger.hh"

#include "detail/NvtxUtils.hh"

namespace celeritas
{
namespace
{
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
