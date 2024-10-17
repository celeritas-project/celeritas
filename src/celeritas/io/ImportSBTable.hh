//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportSBTable.hh
//---------------------------------------------------------------------------//
#pragma once

#include "ImportPhysicsVector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Seltzer Berger differential cross sections for a single element.
 *
 * This 2-dimensional table stores the scaled bremsstrahlung differential cross
 * section [mb]. The x grid is the log energy of the incident particle [MeV],
 * and the y grid is the ratio of the gamma energy to the incident energy.
 */
using ImportSBTable = ImportPhysics2DVector;

//---------------------------------------------------------------------------//
}  // namespace celeritas
