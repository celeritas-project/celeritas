//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GenerateCanonical.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>
#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Helper function to generate a random uniform number
template<class RealType, class Generator>
inline CELER_FUNCTION RealType generate_canonical(Generator& g);

//---------------------------------------------------------------------------//
//! Sample a celeritas::real_type on [0, 1).
template<class Generator>
inline CELER_FUNCTION real_type generate_canonical(Generator& g);

//---------------------------------------------------------------------------//
/*!
 * Generate random numbers in [0, 1).
 *
 * This is essentially an implementation detail; partial specialization can be
 * used to sample using special functions with a given generator.
 */
template<class Generator, class RealType = ::celeritas::real_type>
class GenerateCanonical
{
    static_assert(std::is_floating_point<RealType>::value,
                  "RealType must be float or double");

  public:
    //!@{
    //! Type aliases
    using real_type   = RealType;
    using result_type = real_type;
    //!@}

  public:
    // Constructor
    explicit CELER_FUNCTION GenerateCanonical() {}

    // Sample a random number
    result_type operator()(Generator& rng);
};

} // namespace celeritas

#include "GenerateCanonical.i.hh"
