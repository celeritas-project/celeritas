//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SharedParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/global/CoreParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Shared Celeritas params data.
 */
struct SharedParams
{
    std::shared_ptr<CoreParams> params;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
