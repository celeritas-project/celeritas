//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Utils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Types.hh"

namespace celeritas
{
struct ElementDef;

namespace detail
{
//---------------------------------------------------------------------------//
real_type calc_mass_rad_coeff(const ElementDef& el);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
