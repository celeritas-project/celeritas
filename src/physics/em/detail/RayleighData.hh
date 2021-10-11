//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RayleighData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Rayleigh angular parameters (form factor) for sampling the angular
 * distribution of coherently scattered photon
 */
struct RayleighData
{
    static const unsigned int num_parameters = 9;
    static const unsigned int num_elements   = 100;

    static const real_type angular_parameters[num_parameters][num_elements];
};

} // namespace detail
} // namespace celeritas
