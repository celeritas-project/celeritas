//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LoadXs.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include "PhysicsArrayParams.hh"

namespace demo_interactor
{
//---------------------------------------------------------------------------//

std::shared_ptr<celeritas::PhysicsArrayParams> load_xs();

//---------------------------------------------------------------------------//
} // namespace demo_interactor