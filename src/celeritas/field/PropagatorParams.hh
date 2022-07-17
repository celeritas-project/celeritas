//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/PropagatorParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Array.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Data for propagation in a straight line (none needed).
 */
struct LinearPropagatorParams
{
};

//---------------------------------------------------------------------------//
/*!
 * Data for propagation in a uniform magnetic field.
 */
struct UniformMagPropagatorParams
{
    Real3              field; //!< Field strength [native units]
    FieldDriverOptions driver_options;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
