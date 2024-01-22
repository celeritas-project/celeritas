//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-interactor/LoadXs.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "XsGridParams.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//

std::shared_ptr<XsGridParams> load_xs();

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
