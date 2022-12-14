//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/InitializeActions.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

class G4RunManager;

namespace celeritas
{
struct SetupOptions;
struct SharedParams;

//---------------------------------------------------------------------------//
void InitializeActions(const std::shared_ptr<const SetupOptions>& options,
                       const std::shared_ptr<SharedParams>&       params,
                       G4RunManager*                              manager);

//---------------------------------------------------------------------------//
} // namespace celeritas
