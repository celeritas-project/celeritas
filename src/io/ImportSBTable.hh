//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportSBTable.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Seltzer Berger differential cross sections for a single element.
 */
struct ImportSBTable
{
    std::vector<double> x;     //!< Log energy of incident particle / MeV
    std::vector<double> y;     //!< Ratio of gamma energy to incident energy
    std::vector<double> value; //!< Scaled DCS [mb] [ny * x + y]
};

//---------------------------------------------------------------------------//
} // namespace celeritas
