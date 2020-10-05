//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportTableType.hh
//! \brief Enumerator for the physics table types
//---------------------------------------------------------------------------//
#pragma once

#include <string>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * This enum was created to safely access the many imported physics tables.
 */
enum class ImportTableType
{
    // Geant4 table types
    not_defined,
    dedx,
    ionisation,
    range,
    range_sec,
    inverse_range,
    lambda,
    lambda_prim,
    lambda_mod_1,
    lambda_mod_2,
    lambda_mod_3,
    lambda_mod_4
};

//---------------------------------------------------------------------------//
} // namespace celeritas
