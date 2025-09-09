// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/detail/NvtxUtils.hh
// //---------------------------------------------------------------------------//
#pragma once

#include <string_view>
#include <nvtx3/nvToolsExt.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Library-wide handle to the domain name.
nvtxDomainHandle_t domain_handle();

//! Retrieve the handle for a given message.
nvtxStringHandle_t message_handle_for(std::string_view message);

//---------------------------------------------------------------------------//
}  // namespace celeritas
