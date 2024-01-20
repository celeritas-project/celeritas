//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportLivermorePE.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "ImportPhysicsVector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Livermore EPICS2014 photoelectric cross section data for a single element.
 */
struct ImportLivermoreSubshell
{
    double binding_energy;  //!< Ionization energy [MeV]
    std::vector<double> param_lo;  //!< Low energy xs fit parameters
    std::vector<double> param_hi;  //!< High energy xs fit parameters
    std::vector<double> xs;  //!< Tabulated cross sections [b]
    std::vector<double> energy;  //!< Tabulated energies [MeV]
};

struct ImportLivermorePE
{
    ImportPhysicsVector xs_lo;  //!< Low energy range tabulated xs [b]
    ImportPhysicsVector xs_hi;  //!< High energy range tabulated xs [b]
    double thresh_lo;  //!< Threshold for low energy fit [MeV]
    double thresh_hi;  //!< Threshold for high energy fit [MeV]
    std::vector<ImportLivermoreSubshell> shells;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
