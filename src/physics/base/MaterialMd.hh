//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialMd.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include "base/OpaqueId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Host-only metadata for a material.
 */
struct MaterialMd
{
    std::string name; // Material name
};

//---------------------------------------------------------------------------//
} // namespace celeritas
