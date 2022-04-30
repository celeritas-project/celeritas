//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Utils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Types.hh"
#include "physics/base/Units.hh"

namespace celeritas
{
struct ElementRecord;

namespace detail
{
//---------------------------------------------------------------------------//
real_type        calc_coulomb_correction(int atomic_number);
real_type        calc_mass_rad_coeff(const ElementRecord& el);
units::MevEnergy get_mean_excitation_energy(int atomic_number);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
