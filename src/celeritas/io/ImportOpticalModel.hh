//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportOpticalModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/inp/Grid.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Enumerator for available optical physics models.
 *
 * This enum is used to identify the optical model that imported model MFP
 * tables correspond to.
 */
enum class ImportModelClass
{
    absorption,
    rayleigh,
    wls,
    wls2,
    mie,
    size_

};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Get the string form of one of the enumerations
char const* to_cstring(ImportModelClass imc);

//---------------------------------------------------------------------------//
}  // namespace optical

//---------------------------------------------------------------------------//
/*!
 * Imported data for an optical physics model.
 */
struct ImportOpticalModel
{
    optical::ImportModelClass model_class;
    std::vector<inp::Grid> mfp_table;  //!< per optical material MFPs
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
