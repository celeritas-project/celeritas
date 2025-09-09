//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportSBTable.hh
//! \deprecated Remove in v1.0
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/inp/Grid.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Seltzer Berger differential cross sections for a single element.
 *
 * This 2-dimensional table stores the scaled bremsstrahlung differential cross
 * section [mb]. The x grid is the log energy of the incident particle [MeV],
 * and the y grid is the ratio of the gamma energy to the incident energy.
 *
 * DEPRECATED: remove in v1.0.
 */
using ImportSBTable = inp::TwodGrid;

//---------------------------------------------------------------------------//
}  // namespace celeritas
