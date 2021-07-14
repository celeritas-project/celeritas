//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CMSParameterizedField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Evaluate the value of magnetic field using a parameterized function in the
 * tracker volume of the CMS detector. The implementation is based on the
 * TkBfield class of CMSSW (the TOSCA computation version 1103l),
 * https://cmssdt.cern.ch/lxr/source/MagneticField/ParametrizedEngine/src/
 * and used for testing the charged particle propagation in a magnetic field.
 */
class CMSParameterizedField
{
    //!@{
    //! Type aliases
    using Real4 = celeritas::Array<real_type, 4>;
    //!@}

  public:
    CELER_FUNCTION
    inline CMSParameterizedField() {}

    // Return the magnetic field for the given position
    CELER_FUNCTION
    inline Real3 operator()(const Real3& pos) const;

  private:
    // Evaluate the magnetic field for the given r and z
    CELER_FUNCTION
    inline Real3 evaluate_field(real_type r, real_type z) const;

    // Evaluate the parameterized function and its derivatives
    CELER_FUNCTION
    inline Real4 evaluate_parameters(real_type x) const;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "CMSParameterizedField.i.hh"
