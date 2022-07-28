//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/UniformFieldData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

#include "FieldDriverOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

struct UniformFieldParams
{
    Real3              field{0, 0, 0};
    FieldDriverOptions options;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
