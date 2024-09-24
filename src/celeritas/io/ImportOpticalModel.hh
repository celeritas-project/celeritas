//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportOpticalModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "ImportPhysicsVector.hh"

namespace celeritas
{
namespace optical
{
enum class ImportModelClass
{
    other,
    // Optical
    absorption,
    rayleigh,
    wls,
    //
    size_
};

char const* to_cstring(ImportModelClass imc);
}  // namespace optical

//---------------------------------------------------------------------------//
/*!
 */
struct ImportOpticalModel
{
    optical::ImportModelClass model_class;
    std::vector<ImportPhysicsVector> mfps;  //!< per optical material MFPs
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
