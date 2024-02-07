//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/detail/LengthUnits.hh
//! \brief NOTE: only use inside geocel; prefer celeritas/Units.hh
//---------------------------------------------------------------------------//
#pragma once

#define CELER_ICRT inline constexpr real_type

namespace celeritas
{
namespace lengthunits
{
//---------------------------------------------------------------------------//
#if CELERITAS_UNITS == CELERITAS_UNITS_CGS
CELER_ICRT meter = 100;
CELER_ICRT centimeter = 1;
CELER_ICRT millimeter = 0.1;
#elif CELERITAS_UNITS == CELERITAS_UNITS_SI
CELER_ICRT meter = 1;
CELER_ICRT centimeter = 0.01;
CELER_ICRT millimeter = 0.001;
#elif CELERITAS_UNITS == CELERITAS_UNITS_CLHEP
CELER_ICRT meter = 1000;
CELER_ICRT centimeter = 10;
CELER_ICRT millimeter = 1;
#endif

//---------------------------------------------------------------------------//
}  // namespace lengthunits
}  // namespace celeritas
